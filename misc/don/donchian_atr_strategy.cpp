// donchian_atr_strategy.cpp
// Educational, backtest-first example. Not investment advice.
//
// Strategy (daily bars, one liquid equity/ETF at a time):
//   * Enter long when today's close exceeds the highest HIGH of the prior
//     55 completed sessions. The order fills at the NEXT session's open.
//   * Exit when today's close falls below the lowest LOW of the prior
//     20 completed sessions. The order fills at the NEXT session's open.
//   * Also maintain a 3-ATR trailing stop. A gap through the stop fills at
//     the opening price; otherwise a touched stop fills at the stop price.
//   * Size the trade so the initial stop risks 1% of current equity, while
//     limiting gross allocation to 95% of equity. Whole shares only.
//   * Charge configurable slippage and commission on every fill.
//
// Important data rule: Open, High, Low and Close must be consistently
// split/dividend adjusted (or all consistently unadjusted). Mixing adjusted
// Close with raw OHLC creates false signals and returns.
//
// Build:
//   g++ -std=c++17 -O2 -Wall -Wextra -pedantic donchian_atr_strategy.cpp -o strategy
//
// Run:
//   ./strategy daily_ohlcv.csv
//   ./strategy daily_ohlcv.csv 100000 1.0 5.0
//
// Optional arguments after the CSV are:
//   initial_cash, slippage_bps, commission_per_order
//
// Required CSV columns (case/spacing insensitive):
//   Timestamp, Open, High, Low, Close, Volume
// "Date" is also accepted as an alias for "Timestamp". Rows must be
// oldest-to-newest. Supported timestamp formats include:
//   8/31/2026  4:00:00 AM
//   2026-08-31
//   2026-08-31T04:00:00Z

#include <algorithm>
#include <cmath>
#include <cctype>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

struct Bar {
    std::string date;
    long long timestamp_key = 0; // YYYYMMDDhhmmss, used only for ordering
    double open = 0.0;
    double high = 0.0;
    double low = 0.0;
    double close = 0.0;
    double volume = 0.0;
};

struct Config {
    int entry_lookback = 55;
    int exit_lookback = 20;
    int atr_period = 20;
    double atr_stop_multiple = 3.0;
    double risk_fraction = 0.01;
    double max_allocation = 0.95;
    double initial_cash = 100000.0;
    double slippage_bps = 1.0;
    double commission = 5.0;
};

struct Trade {
    std::string entry_date;
    std::string exit_date;
    int shares = 0;
    double entry_fill = 0.0;
    double exit_fill = 0.0;
    double pnl = 0.0;       // net of entry and exit commissions
    std::string reason;
};

struct Position {
    int shares = 0;
    double entry_fill = 0.0;
    double entry_total_cost = 0.0;
    double trailing_stop = 0.0;
    double high_water_close = 0.0;
    std::string entry_date;
};

static std::string trim(std::string s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    const auto last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
}

static std::string normalize_header(std::string s) {
    s = trim(s);
    std::string out;
    for (unsigned char c : s) {
        if (std::isalnum(c)) out.push_back(static_cast<char>(std::tolower(c)));
    }
    return out;
}

// Handles ordinary quoted CSV fields, including doubled quote characters.
static std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    bool quoted = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (c == '"') {
            if (quoted && i + 1 < line.size() && line[i + 1] == '"') {
                field.push_back('"');
                ++i;
            } else {
                quoted = !quoted;
            }
        } else if (c == ',' && !quoted) {
            fields.push_back(trim(field));
            field.clear();
        } else {
            field.push_back(c);
        }
    }
    if (quoted) throw std::runtime_error("Unclosed quote in CSV row");
    fields.push_back(trim(field));
    return fields;
}

static double parse_number(const std::string& text, std::size_t row,
                           const std::string& column) {
    try {
        std::size_t used = 0;
        const double value = std::stod(text, &used);
        if (used != text.size() || !std::isfinite(value)) throw std::invalid_argument("bad");
        return value;
    } catch (...) {
        throw std::runtime_error("Invalid " + column + " at CSV row " +
                                 std::to_string(row) + ": " + text);
    }
}

static std::string collapse_whitespace(const std::string& text) {
    std::string out;
    bool pending_space = false;
    for (unsigned char c : trim(text)) {
        if (std::isspace(c)) {
            pending_space = !out.empty();
        } else {
            if (pending_space) out.push_back(' ');
            out.push_back(static_cast<char>(c));
            pending_space = false;
        }
    }
    return out;
}

static bool valid_calendar_time(const std::tm& t) {
    const int year = t.tm_year + 1900;
    const int month = t.tm_mon + 1;
    if (year < 1900 || year > 9999 || month < 1 || month > 12 ||
        t.tm_hour < 0 || t.tm_hour > 23 || t.tm_min < 0 || t.tm_min > 59 ||
        t.tm_sec < 0 || t.tm_sec > 59) return false;
    static const int month_days[] = {31, 28, 31, 30, 31, 30,
                                     31, 31, 30, 31, 30, 31};
    int days = month_days[month - 1];
    const bool leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
    if (month == 2 && leap) ++days;
    return t.tm_mday >= 1 && t.tm_mday <= days;
}

static long long parse_timestamp_key(const std::string& text, std::size_t row) {
    const std::string value = collapse_whitespace(text);
    // Longer formats come first so a date-only parser cannot accept a prefix.
    const std::vector<std::string> formats = {
        "%m/%d/%Y %I:%M:%S %p",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d"
    };
    for (const auto& format : formats) {
        std::tm t{};
        std::istringstream in(value);
        in >> std::get_time(&t, format.c_str());
        if (in.fail()) continue;
        in >> std::ws;
        if (in.peek() == 'Z' || in.peek() == 'z') {
            in.get();
            in >> std::ws;
        }
        if (in.peek() != std::char_traits<char>::eof() || !valid_calendar_time(t)) continue;
        const long long year = t.tm_year + 1900;
        const long long month = t.tm_mon + 1;
        return year * 10000000000LL + month * 100000000LL +
               static_cast<long long>(t.tm_mday) * 1000000LL +
               static_cast<long long>(t.tm_hour) * 10000LL +
               static_cast<long long>(t.tm_min) * 100LL + t.tm_sec;
    }
    throw std::runtime_error("Unsupported timestamp at CSV row " +
                             std::to_string(row) + ": " + text);
}

static std::vector<Bar> load_bars(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open input file: " + path);

    std::string line;
    if (!std::getline(in, line)) throw std::runtime_error("CSV is empty");
    const auto headers = split_csv(line);
    std::unordered_map<std::string, std::size_t> col;
    for (std::size_t i = 0; i < headers.size(); ++i) col[normalize_header(headers[i])] = i;

    for (const std::string required : {"open", "high", "low", "close", "volume"}) {
        if (!col.count(required)) throw std::runtime_error("Missing CSV column: " + required);
    }
    if (!col.count("timestamp") && !col.count("date"))
        throw std::runtime_error("Missing CSV column: timestamp (or date)");
    const std::size_t timestamp_col = col.count("timestamp") ? col.at("timestamp") : col.at("date");

    std::vector<Bar> bars;
    std::size_t row = 1;
    while (std::getline(in, line)) {
        ++row;
        if (trim(line).empty()) continue;
        const auto f = split_csv(line);
        auto get = [&](const std::string& name) -> const std::string& {
            if (col.at(name) >= f.size())
                throw std::runtime_error("Too few fields at CSV row " + std::to_string(row));
            return f[col.at(name)];
        };
        Bar b;
        if (timestamp_col >= f.size())
            throw std::runtime_error("Too few fields at CSV row " + std::to_string(row));
        b.date = f[timestamp_col];
        b.timestamp_key = parse_timestamp_key(b.date, row);
        b.open = parse_number(get("open"), row, "Open");
        b.high = parse_number(get("high"), row, "High");
        b.low = parse_number(get("low"), row, "Low");
        b.close = parse_number(get("close"), row, "Close");
        b.volume = parse_number(get("volume"), row, "Volume");
        if (b.date.empty() || b.open <= 0 || b.high <= 0 || b.low <= 0 || b.close <= 0 ||
            b.volume < 0 || b.high < std::max({b.open, b.low, b.close}) ||
            b.low > std::min({b.open, b.high, b.close})) {
            throw std::runtime_error("Invalid OHLCV relationship at CSV row " + std::to_string(row));
        }
        if (!bars.empty() && b.timestamp_key <= bars.back().timestamp_key)
            throw std::runtime_error("Timestamps must be strictly increasing; problem at " + b.date);
        bars.push_back(b);
    }
    if (bars.empty()) throw std::runtime_error("CSV has no data rows");
    return bars;
}

static double prior_high(const std::vector<Bar>& bars, std::size_t i, int lookback) {
    double value = -std::numeric_limits<double>::infinity();
    for (std::size_t j = i - static_cast<std::size_t>(lookback); j < i; ++j)
        value = std::max(value, bars[j].high);
    return value;
}

static double prior_low(const std::vector<Bar>& bars, std::size_t i, int lookback) {
    double value = std::numeric_limits<double>::infinity();
    for (std::size_t j = i - static_cast<std::size_t>(lookback); j < i; ++j)
        value = std::min(value, bars[j].low);
    return value;
}

// Simple moving average of True Range. It includes the current completed bar,
// but is only used for the next day's order or next day's active stop.
static double atr(const std::vector<Bar>& bars, std::size_t i, int period) {
    if (i < static_cast<std::size_t>(period)) return std::numeric_limits<double>::quiet_NaN();
    double sum = 0.0;
    for (std::size_t j = i - static_cast<std::size_t>(period) + 1; j <= i; ++j) {
        const double previous_close = bars[j - 1].close;
        const double tr = std::max({bars[j].high - bars[j].low,
                                    std::abs(bars[j].high - previous_close),
                                    std::abs(bars[j].low - previous_close)});
        sum += tr;
    }
    return sum / period;
}

static void validate_config(const Config& c) {
    if (c.entry_lookback < 2 || c.exit_lookback < 2 || c.atr_period < 2)
        throw std::runtime_error("Lookbacks and ATR period must be at least 2");
    if (c.atr_stop_multiple <= 0 || c.risk_fraction <= 0 || c.risk_fraction > 0.10 ||
        c.max_allocation <= 0 || c.max_allocation > 1.0 || c.initial_cash <= 0 ||
        c.slippage_bps < 0 || c.commission < 0)
        throw std::runtime_error("Invalid strategy configuration");
}

static void save_equity_curve(const std::vector<Bar>& bars,
                              const std::vector<double>& equity) {
    std::ofstream out("equity_curve.csv");
    if (!out) throw std::runtime_error("Cannot create equity_curve.csv");
    out << "Timestamp,Equity\n" << std::fixed << std::setprecision(2);
    for (std::size_t i = 0; i < bars.size(); ++i)
        out << bars[i].date << ',' << equity[i] << '\n';
}

static void report(const std::vector<Bar>& bars, const std::vector<double>& equity,
                   const std::vector<Trade>& trades, const Config& c) {
    const double final_equity = equity.back();
    const double total_return = final_equity / c.initial_cash - 1.0;
    const double years = std::max(1.0 / 252.0, (bars.size() - 1) / 252.0);
    const double cagr = std::pow(final_equity / c.initial_cash, 1.0 / years) - 1.0;
    const double buy_hold = bars.back().close / bars.front().close - 1.0;

    std::vector<double> daily_returns;
    for (std::size_t i = 1; i < equity.size(); ++i) {
        if (equity[i - 1] > 0) daily_returns.push_back(equity[i] / equity[i - 1] - 1.0);
    }
    double annual_vol = 0.0, sharpe = 0.0;
    if (daily_returns.size() > 1) {
        const double mean = std::accumulate(daily_returns.begin(), daily_returns.end(), 0.0) /
                            daily_returns.size();
        double ss = 0.0;
        for (double r : daily_returns) ss += (r - mean) * (r - mean);
        const double sd = std::sqrt(ss / (daily_returns.size() - 1));
        annual_vol = sd * std::sqrt(252.0);
        if (sd > 0) sharpe = mean / sd * std::sqrt(252.0); // zero risk-free assumption
    }

    double peak = equity.front(), max_drawdown = 0.0;
    for (double e : equity) {
        peak = std::max(peak, e);
        max_drawdown = std::min(max_drawdown, e / peak - 1.0);
    }

    int wins = 0;
    double gross_profit = 0.0, gross_loss = 0.0;
    for (const auto& t : trades) {
        if (t.pnl > 0) { ++wins; gross_profit += t.pnl; }
        else gross_loss += -t.pnl;
    }
    const double win_rate = trades.empty() ? 0.0 : static_cast<double>(wins) / trades.size();
    const double profit_factor = gross_loss > 0 ? gross_profit / gross_loss
                                                : std::numeric_limits<double>::infinity();

    std::cout << std::fixed << std::setprecision(2)
              << "Period:             " << bars.front().date << " to " << bars.back().date << '\n'
              << "Starting equity:    $" << c.initial_cash << '\n'
              << "Final equity:       $" << final_equity << '\n'
              << "Total return:        " << total_return * 100 << "%\n"
              << "CAGR:                " << cagr * 100 << "%\n"
              << "Annual volatility:   " << annual_vol * 100 << "%\n"
              << "Sharpe (RF=0):        " << sharpe << '\n'
              << "Maximum drawdown:     " << max_drawdown * 100 << "%\n"
              << "Closed trades:        " << trades.size() << '\n'
              << "Win rate:             " << win_rate * 100 << "%\n"
              << "Profit factor:        " << profit_factor << '\n'
              << "Buy/hold close ratio: " << buy_hold * 100 << "% (rough reference only)\n\n";

    std::cout << "Trades\nTimestamp in             Timestamp out            Shares  Entry       Exit        Net P&L     Reason\n";
    for (const auto& t : trades) {
        std::cout << std::left << std::setw(25) << t.entry_date
                  << std::setw(25) << t.exit_date << std::right
                  << std::setw(7) << t.shares
                  << std::setw(12) << t.entry_fill
                  << std::setw(12) << t.exit_fill
                  << std::setw(13) << t.pnl << "  " << t.reason << '\n';
    }
}

static void backtest(const std::vector<Bar>& bars, const Config& c) {
    const double slip = c.slippage_bps / 10000.0;
    double cash = c.initial_cash;
    Position pos;
    std::vector<Trade> trades;
    std::vector<double> equity(bars.size(), c.initial_cash);

    bool enter_next_open = false;
    bool exit_next_open = false;
    double entry_signal_atr = 0.0;

    auto close_position = [&](const Bar& b, double raw_price, const std::string& reason) {
        const double fill = raw_price * (1.0 - slip);
        const double proceeds = pos.shares * fill - c.commission;
        cash += proceeds;
        trades.push_back({pos.entry_date, b.date, pos.shares, pos.entry_fill,
                          fill, proceeds - pos.entry_total_cost, reason});
        pos = Position{};
        exit_next_open = false;
    };

    for (std::size_t i = 0; i < bars.size(); ++i) {
        const Bar& b = bars[i];

        // Orders generated from yesterday's close execute first at today's open.
        if (exit_next_open && pos.shares > 0) close_position(b, b.open, "channel exit");

        if (enter_next_open && pos.shares == 0) {
            const double fill = b.open * (1.0 + slip);
            const double stop_distance = c.atr_stop_multiple * entry_signal_atr;
            const double morning_equity = cash;
            const int risk_shares = static_cast<int>(std::floor(
                (morning_equity * c.risk_fraction) / stop_distance));
            const int allocation_shares = static_cast<int>(std::floor(
                (morning_equity * c.max_allocation - c.commission) / fill));
            const int affordable_shares = static_cast<int>(std::floor(
                (cash - c.commission) / fill));
            const int shares = std::max(0, std::min({risk_shares, allocation_shares,
                                                    affordable_shares}));
            if (shares > 0) {
                const double cost = shares * fill + c.commission;
                cash -= cost;
                pos = {shares, fill, cost, fill - stop_distance, fill, b.date};
            }
            enter_next_open = false;
        }

        // Today's active stop was fixed using information available before today.
        // Gap risk is explicitly modeled: a gap below the stop fills at the open.
        if (pos.shares > 0 && b.low <= pos.trailing_stop) {
            const double raw_stop_fill = (b.open <= pos.trailing_stop) ? b.open : pos.trailing_stop;
            close_position(b, raw_stop_fill, "ATR stop");
        }

        // Mark to market at the close after all fills.
        equity[i] = cash + pos.shares * b.close;

        // Last bar: liquidate so the reported result contains no open-position value.
        if (i + 1 == bars.size()) {
            if (pos.shares > 0) {
                close_position(b, b.close, "end of data");
                equity[i] = cash;
            }
            break;
        }

        const double current_atr = atr(bars, i, c.atr_period);
        if (pos.shares > 0 && std::isfinite(current_atr)) {
            // This update becomes active on the following session, eliminating
            // unknown intrabar ordering between today's high/low observations.
            pos.high_water_close = std::max(pos.high_water_close, b.close);
            pos.trailing_stop = std::max(pos.trailing_stop,
                                         pos.high_water_close - c.atr_stop_multiple * current_atr);

            if (i >= static_cast<std::size_t>(c.exit_lookback) &&
                b.close < prior_low(bars, i, c.exit_lookback)) {
                exit_next_open = true;
            }
        } else if (pos.shares == 0 && !enter_next_open &&
                   i >= static_cast<std::size_t>(std::max(c.entry_lookback, c.atr_period)) &&
                   std::isfinite(current_atr) && current_atr > 0.0 &&
                   b.close > prior_high(bars, i, c.entry_lookback)) {
            enter_next_open = true;
            entry_signal_atr = current_atr;
        }
    }

    save_equity_curve(bars, equity);
    report(bars, equity, trades, c);
    std::cout << "\nWrote equity_curve.csv\n";
}

int main(int argc, char** argv) {
    try {
        if (argc < 2 || argc > 5) {
            std::cerr << "Usage: " << argv[0]
                      << " data.csv [initial_cash] [slippage_bps] [commission]\n";
            return 1;
        }
        Config c;
        if (argc >= 3) c.initial_cash = std::stod(argv[2]);
        if (argc >= 4) c.slippage_bps = std::stod(argv[3]);
        if (argc >= 5) c.commission = std::stod(argv[4]);
        validate_config(c);

        const auto bars = load_bars(argv[1]);
        const std::size_t minimum = static_cast<std::size_t>(
            std::max({c.entry_lookback, c.exit_lookback, c.atr_period}) + 2);
        if (bars.size() < minimum)
            throw std::runtime_error("Not enough rows; need at least " + std::to_string(minimum));
        backtest(bars, c);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 2;
    }
}
