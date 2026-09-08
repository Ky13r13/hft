# include <iostream>

struct MarketData {
    double price;
    
};
// test push comments
int main(){
    MarketData testdata; 
    testdata.price = 150.30;
    std::cout<<testdata.price<<"\n";

}