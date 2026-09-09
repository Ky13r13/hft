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

class MovingAverageStrategy : public Strategy {
    public: 
        Signal onMarketData(const MarketData& data) override
         {} //add mean rev logic
};
// test push comments
int main(){
    MarketData testdata; 
    testdata.price = 150.30;
    std::cout<<testdata.price<<"\n";

};