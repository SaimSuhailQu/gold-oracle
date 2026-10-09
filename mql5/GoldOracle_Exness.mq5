//+------------------------------------------------------------------+
//|                                          GoldOracle_Exness.mq5 |
//| Gold Oracle executor for Exness (XAUUSD)                          |
//|                                                                   |
//| Reads the signal file produced by the Python brain                |
//| (gold_oracle/signal.py -> gold_signal.json) and executes          |
//| XAUUSD trades on the Exness MT5 account with a fixed-fractional   |
//| risk engine. One position at a time.                              |
//|                                                                   |
//|  signal.json schema (flat, written by Python):                    |
//|   { "timestamp_epoch": 1728..., "symbol": "XAUUSD",                |
//|     "p_up": 0.63, "signal": "LONG"/"SHORT"/"FLAT", ... }           |
//|                                                                   |
//| NOTE: MQL5/MetaEditor is not available to the author of this      |
//| file, so it is carefully written but NOT compiled. Compile with   |
//| F7 in MetaEditor and test on DEMO before any live use.            |
//+------------------------------------------------------------------+
#property copyright "Gold Oracle"
#property version   "1.00"

#include <Trade\Trade.mqh>

//--- inputs
input string InpSignalFile          = "gold_signal.json"; // signal file in MQL5/Files
input double InpRiskPercent         = 1.0;                // risk per trade (% of equity)
input double InpMinWinProbability   = 0.80;               // minimum win probability to trade (e.g. 0.80 = 80%)
input bool   InpRequireOrderFlowAudit = true;             // require Order Flow Audit pass before entry
input int    InpSLDollars           = 30;                 // fallback stop loss in USD per oz
input int    InpTPDollars           = 60;                 // fallback take profit in USD per oz
input int    InpMaxSpreadPoints     = 500;                // max spread (points)
input int    InpSignalMaxAgeHrs     = 36;                 // ignore signals older than this
input ulong  InpMagic               = 20261005;           // magic number
input int    InpTimerSeconds        = 60;                 // re-evaluate every N sec

CTrade trade;

//+------------------------------------------------------------------+
//| Tiny flat-JSON extractors (schema is flat; no nested objects)     |
//+------------------------------------------------------------------+
string JsonGetString(const string json, const string key)
{
   string pat = "\"" + key + "\"";
   int p = StringFind(json, pat);
   if(p < 0) return "";
   p = StringFind(json, ":", p + StringLen(pat));
   if(p < 0) return "";
   p++;
   // skip whitespace; value must be a quoted string
   while(p < StringLen(json))
   {
      ushort c = StringGetCharacter(json, p);
      if(c == ' ' || c == '\t' || c == '\r' || c == '\n') p++;
      else break;
   }
   if(p >= StringLen(json) || StringGetCharacter(json, p) != '\"') return "";
   p++;
   int q = StringFind(json, "\"", p);
   if(q < 0) return "";
   return StringSubstr(json, p, q - p);
}

double JsonGetDouble(const string json, const string key)
{
   string pat = "\"" + key + "\"";
   int p = StringFind(json, pat);
   if(p < 0) return 0.0;
   p = StringFind(json, ":", p + StringLen(pat));
   if(p < 0) return 0.0;
   p++;
   int q = p;
   while(q < StringLen(json))
   {
      ushort c = StringGetCharacter(json, q);
      if((c >= '0' && c <= '9') || c == '.' || c == '-' || c == '+' || c == 'e' || c == 'E')
         q++;
      else break;
   }
   if(q <= p) return 0.0;
   return StringToDouble(StringSubstr(json, p, q - p));
}

long JsonGetLong(const string json, const string key)
{
   return (long)JsonGetDouble(json, key);
}

//+------------------------------------------------------------------+
//| Read the whole signal file into a string                         |
//+------------------------------------------------------------------+
bool ReadSignalFile(string &out_json)
{
   int h = FileOpen(InpSignalFile, FILE_READ | FILE_TXT | FILE_ANSI);
   if(h == INVALID_HANDLE)
   {
      Print("GoldOracle: cannot open ", InpSignalFile, " err=", GetLastError());
      return false;
   }
   ulong size = FileSize(h);
   if(size == 0 || size > 65536)
   {
      FileClose(h);
      Print("GoldOracle: signal file has unexpected size: ", size);
      return false;
   }
   out_json = FileReadString(h, (int)size);
   FileClose(h);
   return StringLen(out_json) > 10;
}

//+------------------------------------------------------------------+
//| Position helpers                                                  |
//+------------------------------------------------------------------+
bool HasPosition(const string sym, long &ticket_out, long &type_out)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong tk = PositionGetTicket(i);
      if(tk == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != sym) continue;
      if(PositionGetInteger(POSITION_MAGIC) != (long)InpMagic) continue;
      ticket_out = (long)tk;
      type_out = PositionGetInteger(POSITION_TYPE);
      return true;
   }
   return false;
}

double CalcVolume(const string sym, double sl_distance)
{
   double equity     = AccountInfoDouble(ACCOUNT_EQUITY);
   double risk_money = equity * InpRiskPercent / 100.0;
   double tick_value = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_VALUE);
   double tick_size  = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_SIZE);
   if(tick_value <= 0 || tick_size <= 0 || sl_distance <= 0) return 0.0;

   double vol = risk_money / (sl_distance / tick_size * tick_value);

   double vmin  = SymbolInfoDouble(sym, SYMBOL_VOLUME_MIN);
   double vmax  = SymbolInfoDouble(sym, SYMBOL_VOLUME_MAX);
   double vstep = SymbolInfoDouble(sym, SYMBOL_VOLUME_STEP);
   if(vstep > 0) vol = MathFloor(vol / vstep) * vstep;
   vol = MathMax(vmin, MathMin(vmax, vol));
   // never risk more than 2x the intended fraction due to rounding
   return vol;
}

bool SpreadOK(const string sym)
{
   long spread = SymbolInfoInteger(sym, SYMBOL_SPREAD);
   return spread >= 0 && spread <= InpMaxSpreadPoints;
}

//+------------------------------------------------------------------+
//| Core evaluation (runs on timer)                                   |
//+------------------------------------------------------------------+
void Evaluate()
{
   const string sym = "XAUUSD";
   if(!SymbolInfoInteger(sym, SYMBOL_SELECT))
      SymbolSelect(sym, true);

   double point = SymbolInfoDouble(sym, SYMBOL_POINT);
   int    digits = (int)SymbolInfoInteger(sym, SYMBOL_DIGITS);
   if(point <= 0) { Print("GoldOracle: bad symbol info for ", sym); return; }

   string json;
   if(!ReadSignalFile(json)) { DrawStatus("NO SIGNAL FILE"); return; }

   string sig = JsonGetString(json, "signal");
   double p_up = JsonGetDouble(json, "p_up");
   long   ts   = JsonGetLong(json, "timestamp_epoch");
   long   now  = TimeGMT();
   if(ts <= 0 || now - ts > InpSignalMaxAgeHrs * 3600)
   {
      DrawStatus("STALE SIGNAL");
      return;
   }

   long ticket = 0, ptype = -1;
   bool in_pos = HasPosition(sym, ticket, ptype);

   // FLAT signal (or unknown): close any open oracle position
   if(sig != "LONG" && sig != "SHORT")
   {
      if(in_pos) { trade.PositionClose(ticket); Print("GoldOracle: FLAT signal -> closed #", ticket); }
      DrawStatus(StringFormat("FLAT (p_up=%.2f)", p_up));
      return;
   }

   // Win probability check (must meet or exceed InpMinWinProbability)
   double win_prob = (sig == "LONG") ? p_up : (1.0 - p_up);
   if(win_prob < InpMinWinProbability)
   {
      if(in_pos) { trade.PositionClose(ticket); Print("GoldOracle: closed due to low win prob: ", win_prob); }
      DrawStatus(StringFormat("LOW PROB (prob=%.2f < %.2f)", win_prob, InpMinWinProbability));
      return;
   }

   // Microstructure & Order Flow Audit gate
   if(InpRequireOrderFlowAudit)
   {
      string audit_verdict = JsonGetString(json, "audit_verdict");
      if(StringLen(audit_verdict) > 0 && StringFind(audit_verdict, "AUDIT_PASSED") < 0)
      {
         if(!in_pos)
         {
            DrawStatus(StringFormat("OF AUDIT BLOCKED (%s)", audit_verdict));
            return;
         }
      }
   }

   long want = (sig == "LONG") ? POSITION_TYPE_BUY : POSITION_TYPE_SELL;
   if(in_pos)
   {
      if(ptype == want) { DrawStatus(StringFormat("HOLD %s (p_up=%.2f)", sig, p_up)); return; }
      trade.PositionClose(ticket);
      Print("GoldOracle: reversed, closed #", ticket);
      in_pos = false;
   }

   if(!SpreadOK(sym)) { DrawStatus("SPREAD TOO WIDE"); return; }

   double sl_dist = InpSLDollars;                 // fallback USD per oz
   double tp_dist = InpTPDollars;
   if(InpUseDynamicATR)
   {
      double dyn_sl = JsonGetDouble(json, "sl_dollars");
      double dyn_tp = JsonGetDouble(json, "tp_dollars");
      if(dyn_sl > 10.0) sl_dist = dyn_sl;
      if(dyn_tp > 10.0) tp_dist = dyn_tp;
   }

   double sl_pts  = sl_dist / point;              // in points
   long stops_level = SymbolInfoInteger(sym, SYMBOL_TRADE_STOPS_LEVEL);
   if(sl_pts < stops_level + 10)
   {
      Print("GoldOracle: SL too tight for broker stops level");
      return;
   }

   double vol = CalcVolume(sym, sl_dist);
   if(vol <= 0) { Print("GoldOracle: volume calc failed"); return; }

   double price, sl, tp;
   trade.SetExpertMagicNumber(InpMagic);
   trade.SetDeviationInPoints(30);
   if(want == POSITION_TYPE_BUY)
   {
      price = SymbolInfoDouble(sym, SYMBOL_ASK);
      sl = NormalizeDouble(price - sl_dist, digits);
      tp = NormalizeDouble(price + tp_dist, digits);
      if(trade.Buy(vol, sym, price, sl, tp, "GoldOracle LONG"))
         Print("GoldOracle: BUY ", vol, " @ ", price, " SL=", sl, " TP=", tp, " p_up=", p_up);
      else
         Print("GoldOracle: BUY failed: ", trade.ResultRetcodeDescription());
   }
   else
   {
      price = SymbolInfoDouble(sym, SYMBOL_BID);
      sl = NormalizeDouble(price + sl_dist, digits);
      tp = NormalizeDouble(price - tp_dist, digits);
      if(trade.Sell(vol, sym, price, sl, tp, "GoldOracle SHORT"))
         Print("GoldOracle: SELL ", vol, " @ ", price, " SL=", sl, " TP=", tp, " p_up=", p_up);
      else
         Print("GoldOracle: SELL failed: ", trade.ResultRetcodeDescription());
   }
   DrawStatus(StringFormat("%s (p_up=%.2f)", sig, p_up));
}

void DrawStatus(const string text)
{
   Comment(StringFormat("GOLD ORACLE  |  %s  |  %s", text, TimeToString(TimeGMT(), TIME_DATE | TIME_MINUTES)));
}

//+------------------------------------------------------------------+
//| EA entry points                                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   trade.SetExpertMagicNumber(InpMagic);
   EventSetTimer(InpTimerSeconds);
   Print("GoldOracle_Exness v1.00 started. Signal file: ", InpSignalFile);
   Evaluate();
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   EventKillTimer();
   Comment("");
}

void OnTimer()
{
   Evaluate();
}
//+------------------------------------------------------------------+
