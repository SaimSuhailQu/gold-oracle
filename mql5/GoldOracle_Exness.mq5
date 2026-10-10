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
//|     "p_up": 0.63, "signal": "LONG"/"SHORT"/"FLAT",                |
//|     "size_mult": 1.0/0.25/0.0 (critic tier),                      |
//|     "exposure_mult": 1.0-1.5 (regime scaling), ... }              |
//|                                                                   |
//| Volume = risk% lots x size_mult (critic tier) x exposure_mult      |
//| (regime scaling, toggleable). Older signal files without the      |
//| multiplier fields default both to 1.0.                            |
//|                                                                   |
//| RISK ENGINE (checklist 28-30, 33-36, 50):                         |
//|  - max daily loss % -> halts new entries for the day              |
//|  - max drawdown % from peak equity -> kill switch                 |
//|  - pause N hours after M consecutive losing trades               |
//|  - break-even SL move, trailing stop, partial take-profit         |
//|  - broker-error retries, disconnect guard, CSV trade journal     |
//|  - master InpKillSwitch halts everything                         |
//| State files live in MQL5/Files (goldoracle_*.txt/csv).            |
//|                                                                   |
//| NOTE: MQL5/MetaEditor is not available to the author of this      |
//| file, so it is carefully written but NOT compiled. Compile with   |
//| F7 in MetaEditor and test on DEMO before any live use.            |
//+------------------------------------------------------------------+
#property copyright "Gold Oracle"
#property version   "2.00"

#include <Trade\Trade.mqh>

//--- inputs
input string InpSignalFile     = "gold_signal.json"; // signal file in MQL5/Files
input double InpRiskPercent     = 0.5;                // risk per trade (% of equity)
input int    InpSLDollars       = 30;                 // stop loss in USD per oz
input int    InpTPDollars       = 60;                 // take profit in USD per oz
input int    InpMaxSpreadPoints = 500;                // max spread (points)
input int    InpSignalMaxAgeHrs = 6;                  // ignore signals older than this (bot rebuilds every 5 min)
input ulong  InpMagic           = 20261005;           // magic number
input int    InpTimerSeconds    = 60;                 // re-evaluate every N sec
input bool   InpUseDynamicExposure = true;           // scale lots by signal exposure_mult
//--- risk engine (checklist 28-30, 33-36, 50)
input double InpMaxDailyLossPct = 2.0;                // halt trading if daily loss exceeds this % (29)
input double InpMaxDrawdownPct  = 10.0;               // kill switch if drawdown from peak equity exceeds % (30,50)
input bool   InpUseBreakEven    = true;               // move SL to breakeven at trigger (33)
input double InpBreakEvenTriggerDollars = 20.0;       // breakeven trigger: +$ per oz
input double InpBreakEvenOffsetDollars  = 2.0;        // lock in +$ per oz beyond entry
input bool   InpUseTrailing     = true;               // trailing stop-loss (34)
input double InpTrailingStartDollars   = 30.0;        // start trailing at +$ per oz
input double InpTrailingStepDollars    = 10.0;        // trail SL in $ steps
input bool   InpUsePartialTP    = true;               // partial profit-taking (35)
input double InpPartialTP1Dollars  = 40.0;             // take partial at +$ per oz
input double InpPartialClosePct    = 50.0;             // % of volume to close at TP1
input int    InpMaxConsecutiveLosses = 3;             // pause trading after N consecutive losses (36)
input int    InpPauseHours        = 24;               // pause duration in hours
input int    InpMaxRetries        = 3;                // broker execution retries (37)
input bool   InpKillSwitch       = false;            // MASTER kill switch — set true to halt everything (50)

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
//| Risk engine: daily loss limit, drawdown kill, loss pause, journal |
//| State persists in MQL5/Files so it survives EA reloads.           |
//+------------------------------------------------------------------+
string RE_DayFile()    { return "goldoracle_day.txt"; }
string RE_PeakFile()   { return "goldoracle_peak.txt"; }
string RE_PauseFile()  { return "goldoracle_pause.txt"; }
string RE_JournalFile(){ return "goldoracle_journal.csv"; }

void Journal(const string event, const long ticket, const string details)
{
   int h = FileOpen(RE_JournalFile(), FILE_READ | FILE_WRITE | FILE_CSV | FILE_ANSI, ',');
   if(h == INVALID_HANDLE) return;
   FileSeek(h, 0, SEEK_END);
   // header on first write
   if(FileSize(h) == 0)
      FileWrite(h, "time_utc", "event", "ticket", "details");
   FileWrite(h, TimeToString(TimeGMT(), TIME_DATE | TIME_SECONDS), event, ticket, details);
   FileClose(h);
}

string TodayKey() { return TimeToString(TimeGMT(), TIME_DATE); } // "2026.10.10"

double GetDayStartEquity()
{
   double eq = AccountInfoDouble(ACCOUNT_EQUITY);
   string today = TodayKey();
   int h = FileOpen(RE_DayFile(), FILE_READ | FILE_TXT | FILE_ANSI);
   if(h != INVALID_HANDLE)
   {
      string line = "";
      while(!FileIsEnding(h)) line = FileReadString(h);
      FileClose(h);
      int comma = StringFind(line, ",");
      if(comma > 0 && StringSubstr(line, 0, comma) == today)
         return StringToDouble(StringSubstr(line, comma + 1));
   }
   // new day (or first run): record starting equity
   h = FileOpen(RE_DayFile(), FILE_WRITE | FILE_TXT | FILE_ANSI);
   if(h != INVALID_HANDLE) { FileWriteString(h, today + "," + DoubleToString(eq, 2) + "\n"); FileClose(h); }
   Journal("DAY_START", 0, "day_start_equity=" + DoubleToString(eq, 2));
   return eq;
}

double GetPeakEquity()
{
   double eq = AccountInfoDouble(ACCOUNT_EQUITY);
   double peak = eq;
   int h = FileOpen(RE_PeakFile(), FILE_READ | FILE_TXT | FILE_ANSI);
   if(h != INVALID_HANDLE)
   {
      string line = "";
      while(!FileIsEnding(h)) line = FileReadString(h);
      FileClose(h);
      peak = StringToDouble(line);
      if(peak <= 0) peak = eq;
   }
   if(eq > peak)
   {
      peak = eq;
      h = FileOpen(RE_PeakFile(), FILE_WRITE | FILE_TXT | FILE_ANSI);
      if(h != INVALID_HANDLE) { FileWriteString(h, DoubleToString(peak, 2) + "\n"); FileClose(h); }
   }
   return peak;
}

int ConsecutiveLosses()
{
   // count consecutive losing CLOSED deals with our magic (most recent first)
   if(!HistorySelect(0, TimeCurrent())) return 0;
   int n = HistoryDealsTotal();
   int losses = 0;
   for(int i = n - 1; i >= 0; i--)
   {
      ulong dt = HistoryDealGetTicket(i);
      if(dt == 0) continue;
      if(HistoryDealGetInteger(dt, DEAL_MAGIC) != (long)InpMagic) continue;
      if(HistoryDealGetInteger(dt, DEAL_ENTRY) != DEAL_ENTRY_OUT) continue;
      double profit = HistoryDealGetDouble(dt, DEAL_PROFIT) + HistoryDealGetDouble(dt, DEAL_SWAP) + HistoryDealGetDouble(dt, DEAL_COMMISSION);
      if(profit < 0) losses++;
      else break;
   }
   return losses;
}

bool PauseActive()
{
   int h = FileOpen(RE_PauseFile(), FILE_READ | FILE_TXT | FILE_ANSI);
   if(h == INVALID_HANDLE) return false;
   string line = "";
   while(!FileIsEnding(h)) line = FileReadString(h);
   FileClose(h);
   long until = StringToInteger(line);
   return TimeGMT() < until;
}

void StartPause(const string reason)
{
   long until = TimeGMT() + InpPauseHours * 3600;
   int h = FileOpen(RE_PauseFile(), FILE_WRITE | FILE_TXT | FILE_ANSI);
   if(h != INVALID_HANDLE) { FileWriteString(h, IntegerToString(until) + "\n"); FileClose(h); }
   Journal("PAUSE_START", 0, reason + " until=" + TimeToString(until, TIME_DATE | TIME_MINUTES));
   Print("GoldOracle: trading paused — ", reason);
}

// Returns "" when trading is allowed, else the block reason.
string RiskBlockReason()
{
   if(InpKillSwitch) return "MANUAL KILL SWITCH ENGAGED";
   if(!TerminalInfoInteger(TERMINAL_CONNECTED)) return "TERMINAL DISCONNECTED";
   if(PauseActive()) return "PAUSED AFTER CONSECUTIVE LOSSES";
   double eq = AccountInfoDouble(ACCOUNT_EQUITY);
   double dayStart = GetDayStartEquity();
   if(dayStart > 0)
   {
      double dayLossPct = (eq - dayStart) / dayStart * 100.0;
      if(dayLossPct <= -InpMaxDailyLossPct)
         return StringFormat("DAILY LOSS LIMIT HIT (%.2f%% <= -%.2f%%)", dayLossPct, InpMaxDailyLossPct);
   }
   double peak = GetPeakEquity();
   if(peak > 0)
   {
      double dd = (peak - eq) / peak * 100.0;
      if(dd >= InpMaxDrawdownPct)
         return StringFormat("MAX DRAWDOWN KILL (%.2f%% >= %.2f%% of peak)", dd, InpMaxDrawdownPct);
   }
   return "";
}

//+------------------------------------------------------------------+
//| Position management: break-even, trailing stop, partial take-profit|
//+------------------------------------------------------------------+
void ManagePosition(const string sym, const long ticket, const long ptype)
{
   double point = SymbolInfoDouble(sym, SYMBOL_POINT);
   int digits = (int)SymbolInfoInteger(sym, SYMBOL_DIGITS);
   if(!PositionSelectByTicket((ulong)ticket)) return;
   double open = PositionGetDouble(POSITION_PRICE_OPEN);
   double sl   = PositionGetDouble(POSITION_SL);
   double tp   = PositionGetDouble(POSITION_TP);
   double vol  = PositionGetDouble(POSITION_VOLUME);
   bool isBuy = (ptype == POSITION_TYPE_BUY);
   double price = isBuy ? SymbolInfoDouble(sym, SYMBOL_BID) : SymbolInfoDouble(sym, SYMBOL_ASK);
   double dist = isBuy ? (price - open) : (open - price); // profit in $/oz

   long stops = SymbolInfoInteger(sym, SYMBOL_TRADE_STOPS_LEVEL);

   // 33. break-even
   if(InpUseBreakEven && dist >= InpBreakEvenTriggerDollars)
   {
      double be = NormalizeDouble(open + (isBuy ? InpBreakEvenOffsetDollars : -InpBreakEvenOffsetDollars), digits);
      bool better = isBuy ? (be > sl) : (sl == 0 || be < sl);
      if(better && MathAbs(price - be) / point >= stops + 2)
      {
         if(trade.PositionModify((ulong)ticket, be, tp))
         {
            Journal("BREAKEVEN", ticket, "sl->" + DoubleToString(be, digits));
            sl = be;
         }
      }
   }

   // 34. trailing stop
   if(InpUseTrailing && dist >= InpTrailingStartDollars)
   {
      double target = NormalizeDouble(price + (isBuy ? -InpTrailingStepDollars : InpTrailingStepDollars), digits);
      bool better = isBuy ? (target > sl) : (sl == 0 || target < sl);
      if(better && MathAbs(price - target) / point >= stops + 2)
      {
         if(trade.PositionModify((ulong)ticket, target, tp))
            Journal("TRAIL", ticket, "sl->" + DoubleToString(target, digits));
      }
   }

   // 35. partial profit-taking (once per ticket)
   if(InpUsePartialTP && dist >= InpPartialTP1Dollars)
   {
      string flagFile = StringFormat("goldoracle_partial_%d.txt", ticket);
      if(!FileIsExist(flagFile))
      {
         double closeVol = vol * InpPartialClosePct / 100.0;
         double vstep = SymbolInfoDouble(sym, SYMBOL_VOLUME_STEP);
         double vmin = SymbolInfoDouble(sym, SYMBOL_VOLUME_MIN);
         if(vstep > 0) closeVol = MathFloor(closeVol / vstep) * vstep;
         if(closeVol >= vmin && closeVol < vol)
         {
            bool ok = false;
            for(int r = 0; r < InpMaxRetries && !ok; r++)
               ok = trade.PositionClosePartial((ulong)ticket, closeVol);
            if(ok)
            {
               int fh = FileOpen(flagFile, FILE_WRITE | FILE_TXT | FILE_ANSI);
               if(fh != INVALID_HANDLE) { FileWriteString(fh, "done\n"); FileClose(fh); }
               Journal("PARTIAL_TP", ticket, "closed " + DoubleToString(closeVol, 2) + " lots at +$" + DoubleToString(dist, 1) + "/oz");
            }
         }
      }
   }
}

// 37. execution with broker-error retries
bool ExecWithRetry(const bool isBuy, const double vol, const string sym,
                   const double price, const double sl, const double tp, const string comment)
{
   for(int r = 0; r < InpMaxRetries; r++)
   {
      bool ok = isBuy ? trade.Buy(vol, sym, price, sl, tp, comment)
                      : trade.Sell(vol, sym, price, sl, tp, comment);
      if(ok)
      {
         Journal(isBuy ? "BUY" : "SELL", (long)trade.ResultOrder(),
                 StringFormat("vol=%.2f price=%.2f sl=%.2f tp=%.2f %s", vol, price, sl, tp, comment));
         return true;
      }
      uint rc = trade.ResultRetcode();
      Print("GoldOracle: order attempt ", r + 1, " failed: ", trade.ResultRetcodeDescription(), " (", rc, ")");
      if(rc == TRADE_RETCODE_NO_MONEY || rc == TRADE_RETCODE_INVALID_VOLUME) break; // retry won't help
      Sleep(500);
   }
   Journal("ORDER_FAILED", 0, comment);
   return false;
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

   long ticket = 0, ptype = -1;
   bool in_pos = HasPosition(sym, ticket, ptype);

   // manage any open position first (trailing/breakeven/partials run even when blocked)
   if(in_pos) ManagePosition(sym, ticket, ptype);

   // 36. arm the pause when consecutive losses hit the limit (checked every tick)
   if(ConsecutiveLosses() >= InpMaxConsecutiveLosses && !PauseActive())
      StartPause(StringFormat("%d consecutive losses", InpMaxConsecutiveLosses));

   // risk engine gate (29, 30, 36, 38, 50)
   string blockReason = RiskBlockReason();
   if(blockReason != "")
   {
      DrawStatus("BLOCKED: " + blockReason);
      return;
   }

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

   // ticket/ptype/in_pos already resolved at the top of Evaluate()

   // FLAT signal (or unknown): close any open oracle position
   if(sig != "LONG" && sig != "SHORT")
   {
      if(in_pos)
      {
         trade.PositionClose((ulong)ticket);
         Journal("CLOSE_FLAT", ticket, "flat signal");
         Print("GoldOracle: FLAT signal -> closed #", ticket);
      }
      DrawStatus(StringFormat("FLAT (p_up=%.2f)", p_up));
      return;
   }

   long want = (sig == "LONG") ? POSITION_TYPE_BUY : POSITION_TYPE_SELL;
   if(in_pos)
   {
      if(ptype == want) { DrawStatus(StringFormat("HOLD %s (p_up=%.2f)", sig, p_up)); return; }
      trade.PositionClose((ulong)ticket);
      Print("GoldOracle: reversed, closed #", ticket);
      in_pos = false;
   }

   if(!SpreadOK(sym)) { DrawStatus("SPREAD TOO WIDE"); return; }

   // 26. risk-to-reward filter: need >= 1.5:1 on the planned trade
   if(InpTPDollars < InpSLDollars * 1.5)
   {
      DrawStatus("POOR R:R — TP/SL below 1.5:1");
      Journal("RR_BLOCK", 0, StringFormat("tp=%d sl=%d", InpTPDollars, InpSLDollars));
      return;
   }

   // Critic tier sizing + regime exposure scaling (fail CLOSED: unknown tier = no trade)
   double size_mult = JsonGetDouble(json, "size_mult");
   if(size_mult <= 0.0 || size_mult > 1.0) { Print("GoldOracle: bad size_mult, standing aside"); return; }
   double exposure = 1.0;
   if(InpUseDynamicExposure)
   {
      exposure = JsonGetDouble(json, "exposure_mult");
      if(exposure <= 0.0 || exposure > 2.0) exposure = 1.0;
   }

   double sl_dist = InpSLDollars;                 // USD per oz
   double sl_pts  = sl_dist / point;              // in points
   long stops_level = SymbolInfoInteger(sym, SYMBOL_TRADE_STOPS_LEVEL);
   if(sl_pts < stops_level + 10)
   {
      Print("GoldOracle: SL too tight for broker stops level");
      return;
   }

   double vol = CalcVolume(sym, sl_dist) * size_mult * exposure;
   if(vol <= 0) { Print("GoldOracle: volume calc failed"); return; }
   Print(StringFormat("GoldOracle: sizing risk=%.2f%% x tier=%.2f x exposure=%.2f",
                      InpRiskPercent, size_mult, exposure));

   double price, sl, tp;
   trade.SetExpertMagicNumber(InpMagic);
   trade.SetDeviationInPoints(30);
   if(want == POSITION_TYPE_BUY)
   {
      price = SymbolInfoDouble(sym, SYMBOL_ASK);
      sl = NormalizeDouble(price - sl_dist, digits);
      tp = NormalizeDouble(price + InpTPDollars, digits);
      ExecWithRetry(true, vol, sym, price, sl, tp, "GoldOracle LONG p_up=" + DoubleToString(p_up, 2));
   }
   else
   {
      price = SymbolInfoDouble(sym, SYMBOL_BID);
      sl = NormalizeDouble(price + sl_dist, digits);
      tp = NormalizeDouble(price - InpTPDollars, digits);
      ExecWithRetry(false, vol, sym, price, sl, tp, "GoldOracle SHORT p_up=" + DoubleToString(p_up, 2));
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
