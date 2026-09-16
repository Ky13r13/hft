#include <iostream>
#include <memory>
#include <vector>

struct MarketData {
    double price;
    double volume;
};
//use this for strong types ? I think 
enum class Signal {
    Long, 
    Short, 
    Hold
};

class Strategy {
    public:
        virtual ~Strategy() = default;
        //roughly this needs to get market data and return a signal
        virtual Signal onMarketData(const MarketData& data) = 0;
};

class EMAStrategy : public Strategy {
private:
    int period;
    double multiplier;
    double currentEMA = 0.0;
    bool isInitialized = false;

public:
    // 1. The Constructor: catches the period you pass in and calculates the multiplier
    EMAStrategy(int customPeriod) {
        period = customPeriod;
        multiplier = 2.0 / (period + 1.0);
    }

    Signal onMarketData(const MarketData& data) override {
        if (!isInitialized) {
            currentEMA = data.price;
            isInitialized = true;
            return Signal::Hold;
        }

        // Uses the dynamic multiplier calculated in the constructor
        currentEMA = (data.price - currentEMA) * multiplier + currentEMA;

        if (data.price > currentEMA) {
            return Signal::Long;
        } else {
            return Signal::Short;
        }
    }
};

std::string signalToString(Signal s) {
    switch (s) {
        case Signal::Long:  return "LONG";
        case Signal::Short: return "SHORT";
        case Signal::Hold:  return "HOLD";
    }
    return "UNKNOWN";
}
// test push comments
int main() {
 
    std::unique_ptr<Strategy> fastEMA = std::make_unique<EMAStrategy>(3); // 3-period for quick testing
    std::unique_ptr<Strategy> slowEMA = std::make_unique<EMAStrategy>(5); // 5-period

    std::vector<double> incomingPrices = {100.0, 102.5, 101.0, 104.2, 103.0, 108.5};

    std::cout << "--- Starting Trading Simulation ---\n";

    for (double price : incomingPrices) {
        MarketData data{price, 500.0}; //

        std::cout << "Price: " << data.price << "\t-> ";
        std::cout << "Fast EMA: " << signalToString(fastEMA->onMarketData(data)) << " | ";
        std::cout << "Slow EMA: " << signalToString(slowEMA->onMarketData(data)) << "\n";
    }

    std::cout << "--- Simulation Complete ---\n";
    return 0;
}