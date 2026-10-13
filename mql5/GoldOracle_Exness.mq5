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
input string InpSymbol          = "";                 // trade symbol (empty = current chart symbol)
input string InpSignalFile     = "gold_signal.json"; // signal file in MQL5/Files
input bool   InpScalpMode       = true;               // Mini-Trade / Scalper Mode (fast frequent trades)
input double InpRiskPercent     = 0.5;                // risk per trade (% of equity)
input double InpMaxLotCap       = 0.30;               // HARD CEILING on trade volume (max lots)
input double InpSLDollars       = 2.5;                // stop loss in USD per oz ($2.50)
input double InpTPDollars       = 5.0;                // take profit in USD per oz ($5.00 -> 2:1 R:R)
input double InpProbLongThreshold  = 0.53;            // min P_up required to open/flip LONG
input double InpProbShortThreshold = 0.47;            // max P_up required to open/flip SHORT
input int    InpMaxSpreadPoints = 500;                // max spread (points)
input int    InpSignalMaxAgeHrs = 48;                 // max signal age (hrs)
input ulong  InpMagic           = 20261005;           // magic number
input int    InpTimerSeconds    = 3;                  // re-evaluate every N sec (3s for scalping)
input bool   InpUseDynamicExposure = true;           // scale lots by signal exposure_mult
//--- risk engine (checklist 28-30, 33-36, 50)
input double InpMaxDailyLossPct = 3.0;                // halt trading if daily loss exceeds this %
input double InpMaxDrawdownPct  = 10.0;               // kill switch if drawdown from peak equity exceeds %
input bool   InpUseBreakEven    = true;               // move SL to breakeven at trigger
input double InpBreakEvenTriggerDollars = 2.0;        // breakeven trigger: +$ per oz
input double InpBreakEvenOffsetDollars  = 0.5;        // lock in +$ per oz beyond entry
input bool   InpUseTrailing     = true;               // trailing stop-loss
input double InpTrailingStartDollars   = 2.5;         // start trailing at +$ per oz
input double InpTrailingStepDollars    = 1.0;         // trail SL in $ steps
input bool   InpUsePartialTP    = true;               // partial profit-taking
input double InpPartialTP1Dollars  = 3.5;              // take partial at +$ per oz
input double InpPartialClosePct    = 50.0;             // % of volume to close at TP1
input int    InpTradeCooldownSec = 180;             // cooldown (sec) after trade closes before next entry
input bool   InpFilterSessions   = true;            // enforce London/NY trading hours (07:00-21:00 UTC)
input int    InpSessionStartUTC  = 7;               // session start hour UTC (07:00 London open)
input int    InpSessionEndUTC    = 21;              // session end hour UTC (21:00 NY close)
input int    InpMaxConsecutiveLosses = 5;             // pause trading after N consecutive losses
input int    InpPauseHours        = 4;                // pause duration in hours
input bool   InpResetLossPause    = false;            // set to true to clear consecutive losses pause
input int    InpMaxRetries        = 3;                // broker execution retries
input bool   InpKillSwitch       = false;            // MASTER kill switch — set true to halt everything

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
   // skip whitespace after colon
   while(p < StringLen(json))
   {
      ushort c = StringGetCharacter(json, p);
      if(c == ' ' || c == '\t' || c == '\r' || c == '\n') p++;
      else break;
   }
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
//| Read the whole signal file into a string (with share & retry)    |
//+------------------------------------------------------------------+
bool ReadSignalFile(string &out_json)
{
   out_json = "";
   int h = INVALID_HANDLE;
   // Retry up to 5 times (10ms pause) if python is actively rewriting the file
   for(int attempt = 0; attempt < 5; attempt++)
   {
      h = FileOpen(InpSignalFile, FILE_READ | FILE_TXT | FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_ANSI);
      if(h == INVALID_HANDLE)
         h = FileOpen(InpSignalFile, FILE_READ | FILE_BIN | FILE_SHARE_READ | FILE_SHARE_WRITE);
      if(h != INVALID_HANDLE) break;
      Sleep(10);
   }

   if(h == INVALID_HANDLE)
   {
      return false;
   }
   ulong size = FileSize(h);
   if(size == 0 || size > 131072)
   {
      FileClose(h);
      return false;
   }
   while(!FileIsEnding(h))
   {
      string line = FileReadString(h);
      out_json += line + "\n";
   }
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
   if(InpMaxLotCap > 0.0 && InpMaxLotCap < vmax) vmax = InpMaxLotCap;
   double vstep = SymbolInfoDouble(sym, SYMBOL_VOLUME_STEP);
   if(vstep > 0) vol = MathFloor(vol / vstep) * vstep;
   vol = MathMax(vmin, MathMin(vmax, vol));
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
   if(InpResetLossPause)
   {
      FileDelete(RE_PauseFile());
      return false;
   }
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
   if(InpResetLossPause) return;
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

   // Session hours check (UTC)
   if(InpFilterSessions)
   {
      MqlDateTime dt;
      TimeToStruct(TimeGMT(), dt);
      if(dt.hour < InpSessionStartUTC || dt.hour >= InpSessionEndUTC)
         return StringFormat("OUTSIDE TRADING SESSION (current hour %02d UTC, allowed %02d-%02d)",
                             dt.hour, InpSessionStartUTC, InpSessionEndUTC);
   }

   // Post-trade cooldown check
   if(InpTradeCooldownSec > 0 && HistorySelect(TimeCurrent() - InpTradeCooldownSec, TimeCurrent()))
   {
      int deals = HistoryDealsTotal();
      for(int i = deals - 1; i >= 0; i--)
      {
         ulong dt = HistoryDealGetTicket(i);
         if(dt > 0 && HistoryDealGetInteger(dt, DEAL_MAGIC) == (long)InpMagic && HistoryDealGetInteger(dt, DEAL_ENTRY) == DEAL_ENTRY_OUT)
         {
            datetime closeTime = (datetime)HistoryDealGetInteger(dt, DEAL_TIME);
            if(TimeCurrent() - closeTime < InpTradeCooldownSec)
            {
               long rem = InpTradeCooldownSec - (TimeCurrent() - closeTime);
               return StringFormat("POST-TRADE COOLDOWN (%ds remaining)", rem);
            }
         }
      }
   }

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

// 37. execution with broker-error retries (fetches fresh Bid/Ask on each attempt)
bool ExecWithRetry(const bool isBuy, const double vol, const string sym,
                   const double sl_dist, const double tp_dist, const string comment)
{
   int digits = (int)SymbolInfoInteger(sym, SYMBOL_DIGITS);
   double point = SymbolInfoDouble(sym, SYMBOL_POINT);
   long stops_level = SymbolInfoInteger(sym, SYMBOL_TRADE_STOPS_LEVEL);

   for(int r = 0; r < InpMaxRetries; r++)
   {
      double price = isBuy ? SymbolInfoDouble(sym, SYMBOL_ASK) : SymbolInfoDouble(sym, SYMBOL_BID);
      double actual_sl_dist = sl_dist;
      double sl_pts = actual_sl_dist / (point > 0 ? point : 0.01);
      if(stops_level > 0 && sl_pts < stops_level + 10)
      {
         actual_sl_dist = (stops_level + 20) * point;
      }

      double sl = isBuy ? NormalizeDouble(price - actual_sl_dist, digits)
                        : NormalizeDouble(price + actual_sl_dist, digits);
      double tp = isBuy ? NormalizeDouble(price + tp_dist, digits)
                        : NormalizeDouble(price - tp_dist, digits);

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
   // 1. Symbol resolution
   string sym = (InpSymbol != "") ? InpSymbol : _Symbol;
   if(sym == "") sym = _Symbol;
   SymbolSelect(sym, true);

   double point = SymbolInfoDouble(sym, SYMBOL_POINT);
   int    digits = (int)SymbolInfoInteger(sym, SYMBOL_DIGITS);
   if(point <= 0)
   {
      static datetime last_bad_sym = 0;
      if(TimeCurrent() - last_bad_sym > 30)
      {
         Print("GoldOracle: bad symbol info for ", sym);
         last_bad_sym = TimeCurrent();
      }
      return;
   }

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
      static datetime last_block_log = 0;
      if(TimeCurrent() - last_block_log > 15)
      {
         Print("GoldOracle blocked by risk engine: ", blockReason);
         last_block_log = TimeCurrent();
      }
      DrawStatus("BLOCKED: " + blockReason);
      return;
   }

   string json;
   if(!ReadSignalFile(json))
   {
      DrawStatus("NO SIGNAL FILE");
      return;
   }

   string sig = JsonGetString(json, "signal");
   double p_up = JsonGetDouble(json, "p_up");
   long   ts   = JsonGetLong(json, "timestamp_epoch");
   long   now  = TimeGMT();
   // Stale signal check (bypassed in Strategy Tester since historical simulated time differs from live signal timestamp)
   if(!MQLInfoInteger(MQL_TESTER) && (ts <= 0 || now - ts > InpSignalMaxAgeHrs * 3600))
   {
      static datetime last_stale_log = 0;
      if(TimeCurrent() - last_stale_log > 30)
      {
         Print(StringFormat("GoldOracle: STALE SIGNAL (ts=%d now=%d diff=%d max=%d)",
                            ts, now, now - ts, InpSignalMaxAgeHrs * 3600));
         last_stale_log = TimeCurrent();
      }
      DrawStatus("STALE SIGNAL");
      return;
   }

   // Directional deadband (hysteresis) to prevent spread churn & rapid whip-sawing
   if(InpScalpMode)
   {
      if(p_up >= InpProbLongThreshold)
         sig = "LONG";
      else if(p_up <= InpProbShortThreshold)
         sig = "SHORT";
      else
         sig = "HOLD"; // inside deadband: hold current stance, do not flip
   }

   // If inside deadband (HOLD), manage position and stand aside
   if(sig == "HOLD")
   {
      if(in_pos)
         DrawStatus(StringFormat("HOLDING POSITION (p_up=%.2f in deadband)", p_up));
      else
         DrawStatus(StringFormat("STANDBY (p_up=%.2f in deadband)", p_up));
      return;
   }

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
      if(ptype == want)
      {
         DrawStatus(StringFormat("HOLD %s (p_up=%.2f)", sig, p_up));
         return;
      }
      // Reversal only occurs when signal decisively crosses opposite threshold
      trade.PositionClose((ulong)ticket);
      Print("GoldOracle: strong opposite signal, reversed and closed #", ticket);
      in_pos = false;
   }

   if(!SpreadOK(sym))
   {
      long cur_spread = SymbolInfoInteger(sym, SYMBOL_SPREAD);
      static datetime last_spread_log = 0;
      if(TimeCurrent() - last_spread_log > 15)
      {
         Print(StringFormat("GoldOracle: SPREAD TOO WIDE for %s (current=%d, max=%d)",
                            sym, cur_spread, InpMaxSpreadPoints));
         last_spread_log = TimeCurrent();
      }
      DrawStatus(StringFormat("SPREAD TOO WIDE (%d > %d)", cur_spread, InpMaxSpreadPoints));
      return;
   }

   // 26. risk-to-reward filter: need >= 1.2:1 on the planned trade
   if(InpTPDollars < InpSLDollars * 1.2)
   {
      DrawStatus("POOR R:R — TP/SL below 1.2:1");
      Journal("RR_BLOCK", 0, StringFormat("tp=%.1f sl=%.1f", InpTPDollars, InpSLDollars));
      return;
   }

   // Sizing: In scalp mode, trade active 1.0 tier
   double size_mult = 1.0;
   if(!InpScalpMode)
   {
      size_mult = JsonGetDouble(json, "size_mult");
      if(size_mult <= 0.0 || size_mult > 1.0)
      {
         Print("GoldOracle: bad size_mult, standing aside");
         return;
      }
   }
   double exposure = 1.0;
   if(InpUseDynamicExposure)
   {
      exposure = JsonGetDouble(json, "exposure_mult");
      if(exposure <= 0.0 || exposure > 2.0) exposure = 1.0;
   }

   double sl_dist = InpSLDollars;                 // USD per oz
   double sl_pts  = sl_dist / point;              // in points
   long stops_level = SymbolInfoInteger(sym, SYMBOL_TRADE_STOPS_LEVEL);
   if(stops_level > 0 && sl_pts < stops_level + 10)
   {
      sl_pts = stops_level + 20;
      sl_dist = sl_pts * point;
      Print("GoldOracle: adjusted sl_dist to broker stops level: ", sl_dist);
   }

   double vol = CalcVolume(sym, sl_dist) * size_mult * exposure;
   if(vol <= 0)
   {
      static datetime last_vol_log = 0;
      if(TimeCurrent() - last_vol_log > 15)
      {
         Print(StringFormat("GoldOracle: CalcVolume failed (vol=%.4f) eq=%.2f risk=%.2f sl=%.2f tick_val=%.4f tick_sz=%.4f",
                            vol, AccountInfoDouble(ACCOUNT_EQUITY), InpRiskPercent, sl_dist,
                            SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_VALUE),
                            SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_SIZE)));
         last_vol_log = TimeCurrent();
      }
      return;
   }

   // Check if Algo Trading is allowed
   if(!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED) || !MQLInfoInteger(MQL_TRADE_ALLOWED))
   {
      static datetime last_algo_log = 0;
      if(TimeCurrent() - last_algo_log > 15)
      {
         Print(StringFormat("GoldOracle: Trading not allowed! Terminal Trade Allowed: %d, MQL Trade Allowed: %d. Check Algo Trading button!",
                            TerminalInfoInteger(TERMINAL_TRADE_ALLOWED),
                            MQLInfoInteger(MQL_TRADE_ALLOWED)));
         last_algo_log = TimeCurrent();
      }
      DrawStatus("ALGO TRADING DISABLED IN MT5");
      return;
   }

   Print(StringFormat("GoldOracle: attempting %s on %s vol=%.2f (p_up=%.2f)",
                      sig, sym, vol, p_up));

   double price, sl, tp;
   trade.SetExpertMagicNumber(InpMagic);
   trade.SetDeviationInPoints(30);
   if(want == POSITION_TYPE_BUY)
   {
      ExecWithRetry(true, vol, sym, sl_dist, InpTPDollars, "GoldOracle LONG p_up=" + DoubleToString(p_up, 2));
   }
   else
   {
      ExecWithRetry(false, vol, sym, sl_dist, InpTPDollars, "GoldOracle SHORT p_up=" + DoubleToString(p_up, 2));
   }
   DrawStatus(StringFormat("%s (p_up=%.2f)", sig, p_up));
}

void DrawStatus(const string text)
{
   Comment(StringFormat("GOLD ORACLE SCALPER  |  %s  |  %s", text, TimeToString(TimeGMT(), TIME_DATE | TIME_SECONDS)));
}

//+------------------------------------------------------------------+
//| EA entry points                                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   trade.SetExpertMagicNumber(InpMagic);
   EventSetTimer(InpTimerSeconds);
   Print("GoldOracle_Exness Scalper v2.10 started. Timer=", InpTimerSeconds, "s, Signal file: ", InpSignalFile);
   Evaluate();
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   EventKillTimer();
   Comment("");
}

void OnTick()
{
   Evaluate();
}

void OnTimer()
{
   Evaluate();
}
//+------------------------------------------------------------------+
