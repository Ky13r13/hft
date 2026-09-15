#pragma once

// Concrete strategies. All state is fixed-size; no allocation after construction.
//
// The existing engine contract emits a scalar target position. Accordingly, the
// options-aware strategies below either trade/hedge the underlying or provide a
// directional proxy for expected dealer hedging flow. They do NOT pretend that a
// scalar Signal can represent a straddle, calendar, risk reversal, or other
// multi-leg option position.
//
// Optional data contract (detected at compile time; absent values remain NaN):
//
//   VolSurface:
//     skew_25d()       // put IV - call IV, decimal vol
//     curvature_25d()  // average wing IV - ATM IV, decimal vol
//     front_iv(), back_iv(), vol_of_vol()
//
//   GammaSurface (portfolio/dealer exposures, in consistent engine units):
//     dealer_delta(), dealer_vega(), dealer_theta(), dealer_rho()
//     dealer_vanna(), dealer_charm(), dealer_vomma()
//     dealer_speed(), dealer_color(), dealer_zomma(), dealer_ultima()
//     option_delta_flow(), option_gamma_flow(), option_vega_flow()
//     call_put_volume_ratio(), call_put_premium_ratio()
//
//   Quote (one supported spelling per value is sufficient):
//     bid/bid(), ask/ask(), bid_size/bid_size()/bid_qty/bid_qty(),
//     ask_size/ask_size()/ask_qty/ask_qty()
//
//   Trade:
//     size/size()/quantity/quantity()/qty/qty(); absent size falls back to 1,
//     making trade imbalance count-weighted rather than volume-weighted.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <utility>

#include "qte/strategy.hpp"

namespace qte::strategies {
namespace detail {

#define QTE_OPTIONAL_METHOD_READER(method)                                      \
    template <typename T, typename = void>                                      \
    struct has_method_##method : std::false_type {};                            \
    template <typename T>                                                       \
    struct has_method_##method<                                                 \
        T, std::void_t<decltype(std::declval<const T&>().method())>>             \
        : std::true_type {};                                                    \
    template <typename T>                                                       \
    inline double read_##method(const T& x) noexcept {                          \
        if constexpr (has_method_##method<T>::value)                            \
            return static_cast<double>(x.method());                             \
        return kNaN;                                                            \
    }

QTE_OPTIONAL_METHOD_READER(skew_25d)
QTE_OPTIONAL_METHOD_READER(curvature_25d)
QTE_OPTIONAL_METHOD_READER(front_iv)
QTE_OPTIONAL_METHOD_READER(back_iv)
QTE_OPTIONAL_METHOD_READER(vol_of_vol)
QTE_OPTIONAL_METHOD_READER(dealer_delta)
QTE_OPTIONAL_METHOD_READER(dealer_vega)
QTE_OPTIONAL_METHOD_READER(dealer_theta)
QTE_OPTIONAL_METHOD_READER(dealer_rho)
QTE_OPTIONAL_METHOD_READER(dealer_vanna)
QTE_OPTIONAL_METHOD_READER(dealer_charm)
QTE_OPTIONAL_METHOD_READER(dealer_vomma)
QTE_OPTIONAL_METHOD_READER(dealer_speed)
QTE_OPTIONAL_METHOD_READER(dealer_color)
QTE_OPTIONAL_METHOD_READER(dealer_zomma)
QTE_OPTIONAL_METHOD_READER(dealer_ultima)
QTE_OPTIONAL_METHOD_READER(option_delta_flow)
QTE_OPTIONAL_METHOD_READER(option_gamma_flow)
QTE_OPTIONAL_METHOD_READER(option_vega_flow)
QTE_OPTIONAL_METHOD_READER(call_put_volume_ratio)
QTE_OPTIONAL_METHOD_READER(call_put_premium_ratio)

#undef QTE_OPTIONAL_METHOD_READER

#define QTE_HAS_METHOD(name)                                                    \
    template <typename T, typename = void>                                      \
    struct has_##name##_method : std::false_type {};                            \
    template <typename T>                                                       \
    struct has_##name##_method<                                                 \
        T, std::void_t<decltype(std::declval<const T&>().name())>>               \
        : std::true_type {}

#define QTE_HAS_MEMBER(name)                                                    \
    template <typename T, typename = void>                                      \
    struct has_##name##_member : std::false_type {};                            \
    template <typename T>                                                       \
    struct has_##name##_member<                                                 \
        T, std::void_t<decltype(std::declval<const T&>().name)>>                 \
        : std::true_type {}

QTE_HAS_METHOD(bid);
QTE_HAS_MEMBER(bid);
QTE_HAS_METHOD(ask);
QTE_HAS_MEMBER(ask);
QTE_HAS_METHOD(bid_size);
QTE_HAS_MEMBER(bid_size);
QTE_HAS_METHOD(ask_size);
QTE_HAS_MEMBER(ask_size);
QTE_HAS_METHOD(bid_qty);
QTE_HAS_MEMBER(bid_qty);
QTE_HAS_METHOD(ask_qty);
QTE_HAS_MEMBER(ask_qty);
QTE_HAS_METHOD(size);
QTE_HAS_MEMBER(size);
QTE_HAS_METHOD(quantity);
QTE_HAS_MEMBER(quantity);
QTE_HAS_METHOD(qty);
QTE_HAS_MEMBER(qty);

#undef QTE_HAS_METHOD
#undef QTE_HAS_MEMBER

template <typename Q>
inline double quote_bid(const Q& q) noexcept {
    if constexpr (has_bid_method<Q>::value) return static_cast<double>(q.bid());
    if constexpr (has_bid_member<Q>::value) return static_cast<double>(q.bid);
    return kNaN;
}

template <typename Q>
inline double quote_ask(const Q& q) noexcept {
    if constexpr (has_ask_method<Q>::value) return static_cast<double>(q.ask());
    if constexpr (has_ask_member<Q>::value) return static_cast<double>(q.ask);
    return kNaN;
}

template <typename Q>
inline double quote_bid_size(const Q& q) noexcept {
    if constexpr (has_bid_size_method<Q>::value)
        return static_cast<double>(q.bid_size());
    if constexpr (has_bid_size_member<Q>::value)
        return static_cast<double>(q.bid_size);
    if constexpr (has_bid_qty_method<Q>::value)
        return static_cast<double>(q.bid_qty());
    if constexpr (has_bid_qty_member<Q>::value)
        return static_cast<double>(q.bid_qty);
    return kNaN;
}

template <typename Q>
inline double quote_ask_size(const Q& q) noexcept {
    if constexpr (has_ask_size_method<Q>::value)
        return static_cast<double>(q.ask_size());
    if constexpr (has_ask_size_member<Q>::value)
        return static_cast<double>(q.ask_size);
    if constexpr (has_ask_qty_method<Q>::value)
        return static_cast<double>(q.ask_qty());
    if constexpr (has_ask_qty_member<Q>::value)
        return static_cast<double>(q.ask_qty);
    return kNaN;
}

template <typename T>
inline double trade_size(const T& t) noexcept {
    double x = kNaN;
    if constexpr (has_size_method<T>::value) x = static_cast<double>(t.size());
    else if constexpr (has_size_member<T>::value) x = static_cast<double>(t.size);
    else if constexpr (has_quantity_method<T>::value)
        x = static_cast<double>(t.quantity());
    else if constexpr (has_quantity_member<T>::value)
        x = static_cast<double>(t.quantity);
    else if constexpr (has_qty_method<T>::value) x = static_cast<double>(t.qty());
    else if constexpr (has_qty_member<T>::value) x = static_cast<double>(t.qty);
    return std::isfinite(x) && x > 0.0 ? x : 1.0;
}

inline double ewma(double old, double x, double alpha) noexcept {
    return std::isfinite(old) ? old + alpha * (x - old) : x;
}

inline double signed_unit(double x) noexcept {
    return x > 0.0 ? 1.0 : (x < 0.0 ? -1.0 : 0.0);
}

inline double positive_or(double x, double fallback) noexcept {
    return std::isfinite(x) && x > 0.0 ? x : fallback;
}

inline uint64_t event_count(double x, uint64_t fallback) noexcept {
    return std::isfinite(x) && x >= 0.0 ? static_cast<uint64_t>(x) : fallback;
}

}  // namespace detail

struct GreekSnapshot {
    double delta = kNaN;
    double gamma = kNaN;
    double vega = kNaN;
    double theta = kNaN;
    double rho = kNaN;
    double vanna = kNaN;  // d(delta)/d(vol), per 1.00 absolute vol
    double charm = kNaN;  // d(delta)/d(day), by convention in this header
    double vomma = kNaN;  // d(vega)/d(vol)
    double speed = kNaN;   // d(gamma)/d(spot)
    double color = kNaN;   // d(gamma)/d(day)
    double zomma = kNaN;   // d(gamma)/d(vol)
    double ultima = kNaN;  // d(vomma)/d(vol)
};

struct SurfaceSnapshot {
    double atm_iv = kNaN;
    double skew_25d = kNaN;
    double curvature_25d = kNaN;
    double front_iv = kNaN;
    double back_iv = kNaN;
    double term_slope = kNaN;
    double vol_of_vol = kNaN;
    double iv_change = kNaN;
    double flip = kNaN;
    double wall_above = kNaN;
    double wall_below = kNaN;
};

struct OrderFlowSnapshot {
    double trade_imbalance = kNaN;   // signed aggressive volume / total volume
    double book_imbalance = kNaN;    // (bid size - ask size) / total size
    double ofi = kNaN;               // normalized best-level order-flow imbalance
    double microprice_bias = kNaN;   // normalized microprice displacement
    double spread_bps = kNaN;
    uint64_t trade_events = 0;
    uint64_t quote_events = 0;
};

struct OptionsFlowSnapshot {
    // Signed changes in dealer option exposure caused by classified option
    // trades. The downstream hedge-pressure convention is -option_delta_flow.
    double option_delta_flow = kNaN;
    double option_gamma_flow = kNaN;
    double option_vega_flow = kNaN;
    double call_put_volume_ratio = kNaN;
    double call_put_premium_ratio = kNaN;
};

// Common base: fill bookkeeping, surface/Greek cache, and fixed-state order flow.
class BaseStrategy : public Strategy {
public:
    void on_market_data(const MarketData& d) noexcept override {
        last_ts_ = d.ts();

        if (d.kind == MarketData::Kind::Quote && d.quote.valid()) {
            const double mid = d.quote.mid();
            if (std::isfinite(mid)) last_mid_ = mid;

            const double bid = detail::quote_bid(d.quote);
            const double ask = detail::quote_ask(d.quote);
            const double bid_sz = detail::quote_bid_size(d.quote);
            const double ask_sz = detail::quote_ask_size(d.quote);

            if (std::isfinite(bid) && std::isfinite(ask) && ask > bid) {
                spread_bps_ = mid > 0.0 ? (ask - bid) / mid * 1e4 : kNaN;

                if (std::isfinite(bid_sz) && std::isfinite(ask_sz) &&
                    bid_sz >= 0.0 && ask_sz >= 0.0 && bid_sz + ask_sz > 0.0) {
                    const double book = (bid_sz - ask_sz) / (bid_sz + ask_sz);
                    book_imbalance_ = detail::ewma(book_imbalance_, book, flow_alpha_);

                    const double micro =
                        (ask * bid_sz + bid * ask_sz) / (bid_sz + ask_sz);
                    const double bias = 2.0 * (micro - mid) / (ask - bid);
                    microprice_bias_ = detail::ewma(
                        microprice_bias_, clampf(bias, -1.0, 1.0), flow_alpha_);

                    if (have_book_) {
                        double raw = 0.0;
                        if (bid >= last_bid_) raw += bid_sz;
                        if (bid <= last_bid_) raw -= last_bid_size_;
                        if (ask <= last_ask_) raw -= ask_sz;
                        if (ask >= last_ask_) raw += last_ask_size_;
                        const double depth = 0.5 *
                            (bid_sz + ask_sz + last_bid_size_ + last_ask_size_);
                        if (depth > 0.0)
                            ofi_ = detail::ewma(
                                ofi_, clampf(raw / depth, -4.0, 4.0), flow_alpha_);
                    }

                    last_bid_ = bid;
                    last_ask_ = ask;
                    last_bid_size_ = bid_sz;
                    last_ask_size_ = ask_sz;
                    have_book_ = true;
                }
            }
            ++quote_events_;
            return;
        }

        if (d.kind == MarketData::Kind::Bar && d.bar.valid()) {
            if (std::isfinite(d.bar.close)) last_mid_ = d.bar.close;
            return;
        }

        if (d.kind == MarketData::Kind::Trade && d.trade.valid()) {
            const double px = d.trade.price;
            if (!std::isfinite(px)) return;

            double sign = 0.0;
            if (std::isfinite(last_mid_)) sign = detail::signed_unit(px - last_mid_);
            if (sign == 0.0 && std::isfinite(last_trade_px_))
                sign = detail::signed_unit(px - last_trade_px_);
            if (sign == 0.0) sign = last_trade_sign_;

            const double qty = detail::trade_size(d.trade);
            signed_trade_ema_ = detail::ewma(
                signed_trade_ema_, sign * qty, flow_alpha_);
            absolute_trade_ema_ = detail::ewma(
                absolute_trade_ema_, qty, flow_alpha_);
            if (absolute_trade_ema_ > 0.0)
                trade_imbalance_ = clampf(
                    signed_trade_ema_ / absolute_trade_ema_, -1.0, 1.0);

            last_trade_px_ = px;
            if (sign != 0.0) last_trade_sign_ = sign;
            ++trade_events_;
        }
    }

    void on_surface_update(const VolSurface& v,
                           const GammaSurface& g) noexcept override {
        const double old_iv = atm_iv_;
        atm_iv_ = v.atm_iv();
        iv_change_ = std::isfinite(old_iv) && std::isfinite(atm_iv_)
                         ? atm_iv_ - old_iv
                         : kNaN;
        term_slope_ = v.term_slope();
        skew_25d_ = detail::read_skew_25d(v);
        curvature_25d_ = detail::read_curvature_25d(v);
        front_iv_ = detail::read_front_iv(v);
        back_iv_ = detail::read_back_iv(v);
        vol_of_vol_ = detail::read_vol_of_vol(v);

        dealer_delta_ = detail::read_dealer_delta(g);
        dealer_gamma_ = g.dealer_gamma();
        dealer_vega_ = detail::read_dealer_vega(g);
        dealer_theta_ = detail::read_dealer_theta(g);
        dealer_rho_ = detail::read_dealer_rho(g);
        dealer_vanna_ = detail::read_dealer_vanna(g);
        dealer_charm_ = detail::read_dealer_charm(g);
        dealer_vomma_ = detail::read_dealer_vomma(g);
        dealer_speed_ = detail::read_dealer_speed(g);
        dealer_color_ = detail::read_dealer_color(g);
        dealer_zomma_ = detail::read_dealer_zomma(g);
        dealer_ultima_ = detail::read_dealer_ultima(g);
        option_delta_flow_ = detail::read_option_delta_flow(g);
        option_gamma_flow_ = detail::read_option_gamma_flow(g);
        option_vega_flow_ = detail::read_option_vega_flow(g);
        call_put_volume_ratio_ = detail::read_call_put_volume_ratio(g);
        call_put_premium_ratio_ = detail::read_call_put_premium_ratio(g);
        flip_ = g.flip_level();
        wall_above_ = g.wall_above();
        wall_below_ = g.wall_below();
        ++surface_updates_;
    }

    void on_fill(const Fill& f) noexcept override {
        ++fills_;
        last_fill_px_ = f.price;
    }

    void reset() noexcept override {
        fills_ = 0;
        last_fill_px_ = kNaN;
        last_ts_ = 0;
        last_mid_ = last_trade_px_ = last_bid_ = last_ask_ = kNaN;
        last_bid_size_ = last_ask_size_ = kNaN;
        last_trade_sign_ = 0.0;
        signed_trade_ema_ = absolute_trade_ema_ = trade_imbalance_ = kNaN;
        book_imbalance_ = ofi_ = microprice_bias_ = spread_bps_ = kNaN;
        trade_events_ = quote_events_ = surface_updates_ = 0;
        have_book_ = false;
        atm_iv_ = iv_change_ = skew_25d_ = curvature_25d_ = kNaN;
        front_iv_ = back_iv_ = term_slope_ = vol_of_vol_ = kNaN;
        dealer_delta_ = dealer_gamma_ = dealer_vega_ = dealer_theta_ = kNaN;
        dealer_rho_ = dealer_vanna_ = dealer_charm_ = dealer_vomma_ = kNaN;
        dealer_speed_ = dealer_color_ = dealer_zomma_ = dealer_ultima_ = kNaN;
        option_delta_flow_ = option_gamma_flow_ = option_vega_flow_ = kNaN;
        call_put_volume_ratio_ = call_put_premium_ratio_ = kNaN;
        flip_ = wall_above_ = wall_below_ = kNaN;
    }

    uint64_t fills() const noexcept { return fills_; }

    GreekSnapshot greeks() const noexcept {
        return {dealer_delta_, dealer_gamma_, dealer_vega_, dealer_theta_,
                dealer_rho_, dealer_vanna_, dealer_charm_, dealer_vomma_,
                dealer_speed_, dealer_color_, dealer_zomma_, dealer_ultima_};
    }

    SurfaceSnapshot surface() const noexcept {
        return {atm_iv_, skew_25d_, curvature_25d_, front_iv_, back_iv_,
                term_slope_, vol_of_vol_, iv_change_, flip_, wall_above_,
                wall_below_};
    }

    OrderFlowSnapshot order_flow() const noexcept {
        return {trade_imbalance_, book_imbalance_, ofi_, microprice_bias_,
                spread_bps_, trade_events_, quote_events_};
    }

    OptionsFlowSnapshot options_flow() const noexcept {
        return {option_delta_flow_, option_gamma_flow_, option_vega_flow_,
                call_put_volume_ratio_, call_put_premium_ratio_};
    }

protected:
    static double clampf(double x, double lo, double hi) noexcept {
        return x < lo ? lo : (x > hi ? hi : x);
    }

    double flow_score(double trade_weight, double book_weight,
                      double ofi_weight, double micro_weight) const noexcept {
        double total = 0.0;
        double weight = 0.0;
        const auto add = [&](double value, double w) noexcept {
            if (std::isfinite(value) && w > 0.0) {
                total += w * clampf(value, -1.0, 1.0);
                weight += w;
            }
        };
        add(trade_imbalance_, trade_weight);
        add(book_imbalance_, book_weight);
        add(ofi_, ofi_weight);
        add(microprice_bias_, micro_weight);
        return weight > 0.0 ? total / weight : kNaN;
    }

    static constexpr double flow_alpha_ = 0.12;

    Timestamp last_ts_ = 0;
    uint64_t fills_ = 0;
    double last_fill_px_ = kNaN;
    double last_mid_ = kNaN;

    double atm_iv_ = kNaN;
    double iv_change_ = kNaN;
    double skew_25d_ = kNaN;
    double curvature_25d_ = kNaN;
    double front_iv_ = kNaN;
    double back_iv_ = kNaN;
    double term_slope_ = kNaN;
    double vol_of_vol_ = kNaN;

    double dealer_delta_ = kNaN;
    double dealer_gamma_ = kNaN;
    double dealer_vega_ = kNaN;
    double dealer_theta_ = kNaN;
    double dealer_rho_ = kNaN;
    double dealer_vanna_ = kNaN;
    double dealer_charm_ = kNaN;
    double dealer_vomma_ = kNaN;
    double dealer_speed_ = kNaN;
    double dealer_color_ = kNaN;
    double dealer_zomma_ = kNaN;
    double dealer_ultima_ = kNaN;
    double option_delta_flow_ = kNaN;
    double option_gamma_flow_ = kNaN;
    double option_vega_flow_ = kNaN;
    double call_put_volume_ratio_ = kNaN;
    double call_put_premium_ratio_ = kNaN;
    double flip_ = kNaN;
    double wall_above_ = kNaN;
    double wall_below_ = kNaN;

    double last_trade_px_ = kNaN;
    double last_trade_sign_ = 0.0;
    double signed_trade_ema_ = kNaN;
    double absolute_trade_ema_ = kNaN;
    double trade_imbalance_ = kNaN;
    double book_imbalance_ = kNaN;
    double ofi_ = kNaN;
    double microprice_bias_ = kNaN;
    double spread_bps_ = kNaN;
    double last_bid_ = kNaN;
    double last_ask_ = kNaN;
    double last_bid_size_ = kNaN;
    double last_ask_size_ = kNaN;
    uint64_t trade_events_ = 0;
    uint64_t quote_events_ = 0;
    uint64_t surface_updates_ = 0;
    bool have_book_ = false;
};

// Fade z-score deviations; size proportional to -z, capped.
class MeanReversion final : public BaseStrategy {
public:
    explicit MeanReversion(const ParamMap& p)
        : entry_z_(std::max(0.0, p.get("entry_z", 1.5))),
          exit_z_(std::max(0.0, p.get("exit_z", 0.3))),
          max_z_(detail::positive_or(p.get("max_z", 3.0), 3.0)) {}

    const char* name() const noexcept override { return "mean_reversion"; }

    Signal generate_signal(const Context& c) noexcept override {
        const double z = c.features->zscore;
        if (!std::isfinite(z)) return Signal::none();

        double target = 0.0;
        if (std::fabs(z) >= entry_z_)
            target = -clampf(z / max_z_, -1.0, 1.0) * c.max_position;
        else if (std::fabs(z) > exit_z_)
            target = c.position;

        return make_signal(c, target * c.size_scale,
                           clampf(std::fabs(z) / max_z_, 0.0, 1.0), id);
    }

private:
    double entry_z_, exit_z_, max_z_;
};

// Follow lookback momentum when above threshold.
class Momentum final : public BaseStrategy {
public:
    explicit Momentum(const ParamMap& p)
        : threshold_(std::max(0.0, p.get("threshold", 0.003))),
          full_(detail::positive_or(p.get("full_at", 0.01), 0.01)) {}

    const char* name() const noexcept override { return "momentum"; }

    Signal generate_signal(const Context& c) noexcept override {
        const double m = c.features->momentum;
        if (!std::isfinite(m)) return Signal::none();
        const double target = std::fabs(m) < threshold_
                                  ? 0.0
                                  : clampf(m / full_, -1.0, 1.0) * c.max_position;
        return make_signal(c, target * c.size_scale,
                           clampf(std::fabs(m) / full_, 0.0, 1.0), id);
    }

private:
    double threshold_, full_;
};

// EMA crossover trend following with fixed-size EMA state.
class TrendFollowing final : public BaseStrategy {
public:
    explicit TrendFollowing(const ParamMap& p)
        : fast_a_(2.0 /
                  (detail::positive_or(p.get("fast", 12), 12) + 1.0)),
          slow_a_(2.0 /
                  (detail::positive_or(p.get("slow", 48), 48) + 1.0)),
          band_(detail::positive_or(p.get("band", 0.0005), 0.0005)) {}

    const char* name() const noexcept override { return "trend_following"; }

    void on_market_data(const MarketData& d) noexcept override {
        BaseStrategy::on_market_data(d);
        double px = kNaN;
        if (d.kind == MarketData::Kind::Quote && d.quote.valid())
            px = d.quote.mid();
        else if (d.kind == MarketData::Kind::Bar && d.bar.valid())
            px = d.bar.close;
        else if (d.kind == MarketData::Kind::Trade && d.trade.valid())
            px = d.trade.price;
        if (!std::isfinite(px)) return;
        if (!std::isfinite(fast_)) {
            fast_ = slow_ = px;
            return;
        }
        fast_ += fast_a_ * (px - fast_);
        slow_ += slow_a_ * (px - slow_);
        ++n_;
    }

    Signal generate_signal(const Context& c) noexcept override {
        if (n_ < 10 || !std::isfinite(slow_) || slow_ == 0.0)
            return Signal::none();
        const double d = (fast_ - slow_) / slow_;
        const double target = d > band_ ? c.max_position
                              : d < -band_ ? -c.max_position
                                           : 0.0;
        return make_signal(c, target * c.size_scale,
                           clampf(std::fabs(d) / (4.0 * band_), 0.0, 1.0), id);
    }

    void reset() noexcept override {
        BaseStrategy::reset();
        fast_ = slow_ = kNaN;
        n_ = 0;
    }

private:
    double fast_a_, slow_a_, band_;
    double fast_ = kNaN, slow_ = kNaN;
    int n_ = 0;
};

// Vol expansion: breakout direction when realized vol is rising over implied.
class VolExpansion final : public BaseStrategy {
public:
    explicit VolExpansion(const ParamMap& p)
        : min_rv_iv_(detail::positive_or(
              p.get("min_rv_over_iv", 1.1), 1.1)),
          z_(std::max(0.0, p.get("breakout_z", 1.0))) {}

    const char* name() const noexcept override { return "vol_expansion"; }

    Signal generate_signal(const Context& c) noexcept override {
        const auto& f = *c.features;
        if (!std::isfinite(f.realized_vol) || !std::isfinite(f.atm_iv) ||
            !std::isfinite(f.zscore))
            return Signal::none();
        const double ratio = f.realized_vol / std::max(f.atm_iv, 1e-6);
        if (ratio < min_rv_iv_) return make_signal(c, 0.0, 0.2, id);
        const double dir = f.zscore > z_ ? 1.0 : (f.zscore < -z_ ? -1.0 : 0.0);
        return make_signal(c, dir * c.max_position * c.size_scale,
                           clampf(ratio - 1.0, 0.0, 1.0), id);
    }

private:
    double min_rv_iv_, z_;
};

// Implied rich vs realized: range proxy in the underlying, not a short-vol trade.
class VolCompression final : public BaseStrategy {
public:
    explicit VolCompression(const ParamMap& p)
        : min_iv_rv_(detail::positive_or(
              p.get("min_iv_over_rv", 1.15), 1.15)),
          size_(clampf(p.get("size_frac", 0.5), 0.0, 1.0)) {}

    const char* name() const noexcept override { return "vol_compression"; }

    Signal generate_signal(const Context& c) noexcept override {
        const auto& f = *c.features;
        if (!std::isfinite(f.realized_vol) || !std::isfinite(f.atm_iv) ||
            !std::isfinite(f.mid))
            return Signal::none();
        const double ratio = f.atm_iv / std::max(f.realized_vol, 1e-6);
        if (ratio < min_iv_rv_) return make_signal(c, 0.0, 0.2, id);

        double target = 0.0;
        if (std::isfinite(wall_above_) && std::isfinite(wall_below_) &&
            wall_above_ > wall_below_ && f.mid >= wall_below_ &&
            f.mid <= wall_above_) {
            const double pos =
                (f.mid - wall_below_) / (wall_above_ - wall_below_);
            target = -(pos - 0.5) * 2.0 * size_ * c.max_position;
        } else if (std::isfinite(f.zscore)) {
            target = -clampf(f.zscore / 2.0, -1.0, 1.0) *
                     size_ * c.max_position;
        }
        return make_signal(c, target * c.size_scale,
                           clampf(ratio - 1.0, 0.0, 1.0), id);
    }

private:
    double min_iv_rv_, size_;
};

// Positive dealer gamma proxy: fade extension relative to the gamma flip.
class GammaScalping final : public BaseStrategy {
public:
    explicit GammaScalping(const ParamMap& p)
        : gamma_scale_(detail::positive_or(
              p.get("gamma_scale", 1e8), 1e8)),
          max_dist_(detail::positive_or(p.get("max_dist", 0.02), 0.02)) {}

    const char* name() const noexcept override { return "gamma_scalping"; }

    Signal generate_signal(const Context& c) noexcept override {
        const auto& f = *c.features;
        if (!std::isfinite(f.dealer_gamma) || !std::isfinite(f.dist_to_flip))
            return Signal::none();
        if (f.dealer_gamma <= 0.0) return make_signal(c, 0.0, 0.1, id);
        const double g = clampf(f.dealer_gamma / gamma_scale_, 0.0, 1.0);
        const double dist = clampf(f.dist_to_flip / max_dist_, -1.0, 1.0);
        const double target = -dist * g * c.max_position;
        return make_signal(c, target * c.size_scale, g, id);
    }

private:
    double gamma_scale_, max_dist_;
};

// Positive gamma -> fade standardized moves; negative gamma -> follow momentum.
class DealerGammaRegime final : public BaseStrategy {
public:
    explicit DealerGammaRegime(const ParamMap& p)
        : gamma_scale_(detail::positive_or(
              p.get("gamma_scale", 1e8), 1e8)),
          entry_z_(std::max(0.0, p.get("entry_z", 0.8))),
          full_z_(detail::positive_or(p.get("full_z", 2.5), 2.5)),
          momentum_threshold_(std::max(
              0.0, p.get("momentum_threshold", 0.002))),
          momentum_full_(detail::positive_or(
              p.get("momentum_full", 0.01), 0.01)),
          flip_buffer_(detail::positive_or(
              p.get("flip_buffer", 0.003), 0.003)) {}

    const char* name() const noexcept override { return "dealer_gamma_regime"; }

    Signal generate_signal(const Context& c) noexcept override {
        const auto& f = *c.features;
        if (!std::isfinite(f.dealer_gamma) || !std::isfinite(f.zscore))
            return Signal::none();
        const double strength =
            clampf(std::fabs(f.dealer_gamma) / gamma_scale_, 0.0, 1.0);
        double target = 0.0;
        double trigger = 0.0;

        if (f.dealer_gamma > 0.0) {
            trigger = std::fabs(f.zscore) / full_z_;
            if (std::fabs(f.zscore) >= entry_z_)
                target = -clampf(f.zscore / full_z_, -1.0, 1.0) *
                         strength * c.max_position;
        } else {
            if (!std::isfinite(f.momentum)) return Signal::none();
            trigger = std::fabs(f.momentum) / momentum_full_;
            if (std::fabs(f.momentum) >= momentum_threshold_)
                target = clampf(f.momentum / momentum_full_, -1.0, 1.0) *
                         strength * c.max_position;
        }

        double certainty = 1.0;
        if (std::isfinite(f.dist_to_flip) && flip_buffer_ > 0.0)
            certainty = 0.25 + 0.75 *
                clampf(std::fabs(f.dist_to_flip) / flip_buffer_, 0.0, 1.0);
        return make_signal(c, target * certainty * c.size_scale,
                           clampf(strength * trigger * certainty, 0.0, 1.0), id);
    }

private:
    double gamma_scale_, entry_z_, full_z_;
    double momentum_threshold_, momentum_full_, flip_buffer_;
};

// Fade only inside well-formed gamma walls and only in positive gamma.
class GammaWallReversion final : public BaseStrategy {
public:
    explicit GammaWallReversion(const ParamMap& p)
        : gamma_scale_(detail::positive_or(
              p.get("gamma_scale", 1e8), 1e8)),
          edge_fraction_(clampf(p.get("edge_fraction", 0.25), 0.01, 0.49)),
          size_fraction_(clampf(p.get("size_frac", 0.6), 0.0, 1.0)) {}

    const char* name() const noexcept override { return "gamma_wall_reversion"; }

    Signal generate_signal(const Context& c) noexcept override {
        const auto& f = *c.features;
        if (!std::isfinite(f.mid) || !std::isfinite(f.dealer_gamma) ||
            f.dealer_gamma <= 0.0 || !std::isfinite(wall_above_) ||
            !std::isfinite(wall_below_) || wall_above_ <= wall_below_)
            return Signal::none();
        if (f.mid < wall_below_ || f.mid > wall_above_)
            return make_signal(c, 0.0, 0.1, id);  // breached wall: do not fade

        const double pos =
            (f.mid - wall_below_) / (wall_above_ - wall_below_);
        const double g = clampf(f.dealer_gamma / gamma_scale_, 0.0, 1.0);
        double target = 0.0;
        double edge = 0.0;
        if (pos < edge_fraction_) {
            edge = (edge_fraction_ - pos) / edge_fraction_;
            target = edge * size_fraction_ * g * c.max_position;
        } else if (pos > 1.0 - edge_fraction_) {
            edge = (pos - (1.0 - edge_fraction_)) / edge_fraction_;
            target = -edge * size_fraction_ * g * c.max_position;
        }
        return make_signal(c, target * c.size_scale,
                           clampf(edge * g, 0.0, 1.0), id);
    }

private:
    double gamma_scale_, edge_fraction_, size_fraction_;
};

// Follow moves away from the flip only when gamma is negative and momentum agrees.
class GammaFlipBreakout final : public BaseStrategy {
public:
    explicit GammaFlipBreakout(const ParamMap& p)
        : gamma_scale_(detail::positive_or(
              p.get("gamma_scale", 1e8), 1e8)),
          min_dist_(std::max(0.0, p.get("min_dist", 0.002))),
          full_dist_(detail::positive_or(
              p.get("full_dist", 0.015), 0.015)),
          momentum_threshold_(std::max(
              0.0, p.get("momentum_threshold", 0.002))),
          momentum_full_(detail::positive_or(
              p.get("momentum_full", 0.01), 0.01)) {}

    const char* name() const noexcept override { return "gamma_flip_breakout"; }

    Signal generate_signal(const Context& c) noexcept override {
        const auto& f = *c.features;
        if (!std::isfinite(f.mid) || !std::isfinite(f.momentum) ||
            !std::isfinite(f.dealer_gamma) || !std::isfinite(flip_) ||
            flip_ <= 0.0 || f.dealer_gamma >= 0.0)
            return Signal::none();
        const double dist = (f.mid - flip_) / flip_;
        if (std::fabs(dist) < min_dist_ || dist * f.momentum <= 0.0 ||
            std::fabs(f.momentum) < momentum_threshold_)
            return make_signal(c, 0.0, 0.15, id);

        const double g =
            clampf(std::fabs(f.dealer_gamma) / gamma_scale_, 0.0, 1.0);
        const double d = clampf(std::fabs(dist) / full_dist_, 0.0, 1.0);
        const double m =
            clampf(std::fabs(f.momentum) / momentum_full_, 0.0, 1.0);
        const double strength = g * std::sqrt(d * m);
        const double target = detail::signed_unit(dist) * strength * c.max_position;
        return make_signal(c, target * c.size_scale, strength, id);
    }

private:
    double gamma_scale_, min_dist_, full_dist_;
    double momentum_threshold_, momentum_full_;
};

// Approximate the underlying hedge required for reported dealer delta.
class DeltaHedge final : public BaseStrategy {
public:
    explicit DeltaHedge(const ParamMap& p)
        : delta_scale_(detail::positive_or(
              p.get("delta_scale", 1e6), 1e6)),
          deadband_(std::max(0.0, p.get("deadband", 0.05))),
          size_fraction_(clampf(p.get("size_frac", 1.0), 0.0, 1.0)) {}

    const char* name() const noexcept override { return "delta_hedge"; }

    Signal generate_signal(const Context& c) noexcept override {
        if (!std::isfinite(dealer_delta_) || delta_scale_ <= 0.0)
            return Signal::none();
        double hedge = -dealer_delta_ / delta_scale_;
        if (std::fabs(hedge) < deadband_) hedge = 0.0;
        hedge = clampf(hedge, -1.0, 1.0) * size_fraction_;
        return make_signal(c, hedge * c.max_position * c.size_scale,
                           clampf(std::fabs(hedge), 0.0, 1.0), id);
    }

private:
    double delta_scale_, deadband_, size_fraction_;
};

// Project hedge flow from vanna (observed IV change) and charm (time passage).
class VannaCharmFlow final : public BaseStrategy {
public:
    explicit VannaCharmFlow(const ParamMap& p)
        : vanna_scale_(detail::positive_or(
              p.get("vanna_scale", 1e7), 1e7)),
          charm_scale_(detail::positive_or(
              p.get("charm_scale", 1e6), 1e6)),
          charm_horizon_days_(std::max(
              0.0, p.get("charm_horizon_days", 1.0 / 24.0))),
          max_iv_change_(detail::positive_or(
              p.get("max_iv_change", 0.03), 0.03)),
          threshold_(std::max(0.0, p.get("threshold", 0.05))),
          sign_multiplier_(p.get("sign_multiplier", 1.0)) {}

    const char* name() const noexcept override { return "vanna_charm_flow"; }

    Signal generate_signal(const Context& c) noexcept override {
        if (surface_updates_ < 2) return Signal::none();
        double projected_delta_change = 0.0;
        double information = 0.0;

        if (std::isfinite(dealer_vanna_) && std::isfinite(iv_change_) &&
            vanna_scale_ > 0.0) {
            const double dsigma = clampf(
                iv_change_, -max_iv_change_, max_iv_change_);
            projected_delta_change += dealer_vanna_ * dsigma / vanna_scale_;
            information += 1.0;
        }
        if (std::isfinite(dealer_charm_) && charm_scale_ > 0.0) {
            projected_delta_change +=
                dealer_charm_ * charm_horizon_days_ / charm_scale_;
            information += 1.0;
        }
        if (information == 0.0) return Signal::none();

        double pressure = -sign_multiplier_ * projected_delta_change;
        if (std::fabs(pressure) < threshold_) pressure = 0.0;
        pressure = clampf(pressure, -1.0, 1.0);
        return make_signal(c, pressure * c.max_position * c.size_scale,
                           clampf(std::fabs(pressure) * information / 2.0,
                                  0.0, 1.0), id);
    }

private:
    double vanna_scale_, charm_scale_, charm_horizon_days_;
    double max_iv_change_, threshold_, sign_multiplier_;
};

// Translate newly traded option delta into the opposite underlying hedge. This
// requires side-classified options trades; open interest or call/put volume alone
// is not a substitute for signed delta flow.
class OptionsFlowHedgePressure final : public BaseStrategy {
public:
    explicit OptionsFlowHedgePressure(const ParamMap& p)
        : delta_flow_scale_(detail::positive_or(
              p.get("delta_flow_scale", 1e6), 1e6)),
          gamma_flow_scale_(detail::positive_or(
              p.get("gamma_flow_scale", 1e7), 1e7)),
          threshold_(std::max(0.0, p.get("threshold", 0.05))),
          size_fraction_(clampf(p.get("size_frac", 0.75), 0.0, 1.0)),
          sign_multiplier_(p.get("sign_multiplier", 1.0)) {}

    const char* name() const noexcept override {
        return "options_flow_hedge_pressure";
    }

    Signal generate_signal(const Context& c) noexcept override {
        if (!std::isfinite(option_delta_flow_)) return Signal::none();

        // A positive change in dealer option delta conventionally requires an
        // offsetting sale of the underlying. Gamma flow modulates urgency but
        // never supplies direction by itself.
        double pressure = -sign_multiplier_ *
                          option_delta_flow_ / delta_flow_scale_;
        double urgency = 1.0;
        if (std::isfinite(option_gamma_flow_))
            urgency += 0.5 * clampf(
                std::fabs(option_gamma_flow_) / gamma_flow_scale_, 0.0, 1.0);
        pressure *= urgency;

        if (std::fabs(pressure) < threshold_) pressure = 0.0;
        pressure = clampf(pressure, -1.0, 1.0) * size_fraction_;
        return make_signal(c, pressure * c.max_position * c.size_scale,
                           clampf(std::fabs(pressure), 0.0, 1.0), id);
    }

private:
    double delta_flow_scale_, gamma_flow_scale_, threshold_;
    double size_fraction_, sign_multiplier_;
};

// Directional risk overlay from put-skew and volatility-term-structure stress.
// Convention: skew_25d = put IV - call IV; term_slope > 0 = back IV - front IV.
class SurfaceStress final : public BaseStrategy {
public:
    explicit SurfaceStress(const ParamMap& p)
        : skew_reference_(p.get("skew_reference", 0.04)),
          skew_scale_(detail::positive_or(p.get("skew_scale", 0.04), 0.04)),
          term_scale_(detail::positive_or(p.get("term_scale", 0.03), 0.03)),
          curvature_reference_(p.get("curvature_reference", 0.015)),
          curvature_scale_(detail::positive_or(
              p.get("curvature_scale", 0.03), 0.03)),
          vol_of_vol_reference_(p.get("vol_of_vol_reference", 0.80)),
          vol_of_vol_scale_(detail::positive_or(
              p.get("vol_of_vol_scale", 0.80), 0.80)),
          trigger_(clampf(p.get("trigger", 0.5), 0.0, 0.99)),
          size_fraction_(clampf(p.get("size_frac", 0.5), 0.0, 1.0)) {}

    const char* name() const noexcept override { return "surface_stress"; }

    Signal generate_signal(const Context& c) noexcept override {
        double score = 0.0;
        double weight = 0.0;
        if (std::isfinite(skew_25d_) && skew_scale_ > 0.0) {
            score += clampf((skew_25d_ - skew_reference_) / skew_scale_,
                            -1.0, 1.0);
            weight += 1.0;
        }
        if (std::isfinite(term_slope_) && term_scale_ > 0.0) {
            score += clampf(-term_slope_ / term_scale_, -1.0, 1.0);
            weight += 1.0;
        }
        if (std::isfinite(curvature_25d_)) {
            score += 0.5 * clampf(
                (curvature_25d_ - curvature_reference_) / curvature_scale_,
                -1.0, 1.0);
            weight += 0.5;
        }
        if (std::isfinite(vol_of_vol_)) {
            score += 0.5 * clampf(
                (vol_of_vol_ - vol_of_vol_reference_) / vol_of_vol_scale_,
                -1.0, 1.0);
            weight += 0.5;
        }
        if (weight == 0.0) return Signal::none();
        score /= weight;
        const double risk = score > trigger_
                                ? (score - trigger_) / (1.0 - trigger_)
                                : 0.0;
        const double target = -clampf(risk, 0.0, 1.0) *
                              size_fraction_ * c.max_position;
        return make_signal(c, target * c.size_scale,
                           clampf(std::fabs(score), 0.0, 1.0), id);
    }

private:
    double skew_reference_, skew_scale_, term_scale_;
    double curvature_reference_, curvature_scale_;
    double vol_of_vol_reference_, vol_of_vol_scale_;
    double trigger_, size_fraction_;
};

// Composite tick-rule trade imbalance, top-of-book imbalance, OFI, microprice.
class OrderFlowMomentum final : public BaseStrategy {
public:
    explicit OrderFlowMomentum(const ParamMap& p)
        : trade_weight_(std::max(0.0, p.get("trade_weight", 0.35))),
          book_weight_(std::max(0.0, p.get("book_weight", 0.20))),
          ofi_weight_(std::max(0.0, p.get("ofi_weight", 0.30))),
          micro_weight_(std::max(0.0, p.get("micro_weight", 0.15))),
          threshold_(std::max(0.0, p.get("threshold", 0.15))),
          full_(detail::positive_or(p.get("full_at", 0.65), 0.65)),
          max_spread_bps_(detail::positive_or(
              p.get("max_spread_bps", 10.0), 10.0)),
          min_events_(detail::event_count(p.get("min_events", 20), 20)) {}

    const char* name() const noexcept override { return "order_flow_momentum"; }

    Signal generate_signal(const Context& c) noexcept override {
        if (trade_events_ + quote_events_ < min_events_) return Signal::none();
        if (std::isfinite(spread_bps_) && spread_bps_ > max_spread_bps_)
            return make_signal(c, 0.0, 0.1, id);
        const double score = flow_score(
            trade_weight_, book_weight_, ofi_weight_, micro_weight_);
        if (!std::isfinite(score)) return Signal::none();
        const double target = std::fabs(score) < threshold_
                                  ? 0.0
                                  : clampf(score / full_, -1.0, 1.0) *
                                        c.max_position;
        return make_signal(c, target * c.size_scale,
                           clampf(std::fabs(score) / full_, 0.0, 1.0), id);
    }

private:
    double trade_weight_, book_weight_, ofi_weight_, micro_weight_;
    double threshold_, full_, max_spread_bps_;
    uint64_t min_events_;
};

// Require price momentum and order flow to agree before entering a breakout.
class FlowConfirmedBreakout final : public BaseStrategy {
public:
    explicit FlowConfirmedBreakout(const ParamMap& p)
        : flow_threshold_(std::max(0.0, p.get("flow_threshold", 0.20))),
          flow_full_(detail::positive_or(p.get("flow_full", 0.65), 0.65)),
          momentum_threshold_(std::max(
              0.0, p.get("momentum_threshold", 0.002))),
          momentum_full_(detail::positive_or(
              p.get("momentum_full", 0.01), 0.01)),
          z_threshold_(std::max(0.0, p.get("z_threshold", 1.0))),
          min_events_(detail::event_count(p.get("min_events", 20), 20)) {}

    const char* name() const noexcept override { return "flow_confirmed_breakout"; }

    Signal generate_signal(const Context& c) noexcept override {
        const auto& f = *c.features;
        if (trade_events_ + quote_events_ < min_events_ ||
            !std::isfinite(f.momentum) || !std::isfinite(f.zscore))
            return Signal::none();
        const double flow = flow_score(0.35, 0.20, 0.30, 0.15);
        if (!std::isfinite(flow) || std::fabs(flow) < flow_threshold_ ||
            std::fabs(f.momentum) < momentum_threshold_ ||
            std::fabs(f.zscore) < z_threshold_ || flow * f.momentum <= 0.0 ||
            flow * f.zscore <= 0.0)
            return make_signal(c, 0.0, 0.15, id);

        const double fs = clampf(std::fabs(flow) / flow_full_, 0.0, 1.0);
        const double ms =
            clampf(std::fabs(f.momentum) / momentum_full_, 0.0, 1.0);
        const double strength = std::sqrt(fs * ms);
        const double target = detail::signed_unit(flow) * strength * c.max_position;
        return make_signal(c, target * c.size_scale, strength, id);
    }

private:
    double flow_threshold_, flow_full_, momentum_threshold_, momentum_full_;
    double z_threshold_;
    uint64_t min_events_;
};

// Gamma controls whether order flow is faded (positive) or followed (negative).
class GammaFlowConfluence final : public BaseStrategy {
public:
    explicit GammaFlowConfluence(const ParamMap& p)
        : gamma_scale_(detail::positive_or(
              p.get("gamma_scale", 1e8), 1e8)),
          flow_threshold_(std::max(0.0, p.get("flow_threshold", 0.18))),
          flow_full_(detail::positive_or(p.get("flow_full", 0.65), 0.65)),
          positive_gamma_size_(clampf(
              p.get("positive_gamma_size", 0.5), 0.0, 1.0)),
          min_events_(detail::event_count(p.get("min_events", 20), 20)) {}

    const char* name() const noexcept override { return "gamma_flow_confluence"; }

    Signal generate_signal(const Context& c) noexcept override {
        const auto& f = *c.features;
        if (trade_events_ + quote_events_ < min_events_ ||
            !std::isfinite(f.dealer_gamma))
            return Signal::none();
        const double flow = flow_score(0.35, 0.20, 0.30, 0.15);
        if (!std::isfinite(flow)) return Signal::none();
        if (std::fabs(flow) < flow_threshold_)
            return make_signal(c, 0.0, 0.1, id);

        const double g =
            clampf(std::fabs(f.dealer_gamma) / gamma_scale_, 0.0, 1.0);
        const double fs = clampf(std::fabs(flow) / flow_full_, 0.0, 1.0);
        const double regime_sign = f.dealer_gamma > 0.0 ? -1.0 : 1.0;
        const double regime_size =
            f.dealer_gamma > 0.0 ? positive_gamma_size_ : 1.0;
        const double target = regime_sign * detail::signed_unit(flow) *
                              fs * g * regime_size * c.max_position;
        return make_signal(c, target * c.size_scale,
                           clampf(fs * g, 0.0, 1.0), id);
    }

private:
    double gamma_scale_, flow_threshold_, flow_full_, positive_gamma_size_;
    uint64_t min_events_;
};

// Joint regime selector: expansion/negative gamma follows; compression/positive
// gamma fades. This is often safer than running both legs independently.
class VolGammaRegime final : public BaseStrategy {
public:
    explicit VolGammaRegime(const ParamMap& p)
        : expansion_ratio_(detail::positive_or(
              p.get("expansion_ratio", 1.10), 1.10)),
          compression_ratio_(detail::positive_or(
              p.get("compression_ratio", 1.15), 1.15)),
          gamma_scale_(detail::positive_or(
              p.get("gamma_scale", 1e8), 1e8)),
          momentum_full_(detail::positive_or(
              p.get("momentum_full", 0.01), 0.01)),
          z_full_(detail::positive_or(p.get("z_full", 2.5), 2.5)) {}

    const char* name() const noexcept override { return "vol_gamma_regime"; }

    Signal generate_signal(const Context& c) noexcept override {
        const auto& f = *c.features;
        if (!std::isfinite(f.realized_vol) || !std::isfinite(f.atm_iv) ||
            !std::isfinite(f.dealer_gamma) || !std::isfinite(f.momentum) ||
            !std::isfinite(f.zscore))
            return Signal::none();
        const double rv_iv = f.realized_vol / std::max(f.atm_iv, 1e-6);
        const double iv_rv = f.atm_iv / std::max(f.realized_vol, 1e-6);
        const double g =
            clampf(std::fabs(f.dealer_gamma) / gamma_scale_, 0.0, 1.0);
        double target = 0.0;
        double confidence = 0.0;

        if (rv_iv >= expansion_ratio_ && f.dealer_gamma < 0.0) {
            const double m = clampf(f.momentum / momentum_full_, -1.0, 1.0);
            target = m * g * c.max_position;
            confidence = std::fabs(m) * g;
        } else if (iv_rv >= compression_ratio_ && f.dealer_gamma > 0.0) {
            const double z = clampf(f.zscore / z_full_, -1.0, 1.0);
            target = -z * g * c.max_position;
            confidence = std::fabs(z) * g;
        }
        return make_signal(c, target * c.size_scale,
                           clampf(confidence, 0.0, 1.0), id);
    }

private:
    double expansion_ratio_, compression_ratio_, gamma_scale_;
    double momentum_full_, z_full_;
};

// Explicit registration entry point. Update its implementation to register:
// dealer_gamma_regime, gamma_wall_reversion, gamma_flip_breakout, delta_hedge,
// vanna_charm_flow, options_flow_hedge_pressure, surface_stress, order_flow_momentum,
// flow_confirmed_breakout, gamma_flow_confluence, and vol_gamma_regime.
void register_all();

}  // namespace qte::strategies
