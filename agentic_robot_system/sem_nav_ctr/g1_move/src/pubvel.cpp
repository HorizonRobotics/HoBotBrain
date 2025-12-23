#include <unitree/robot/g1/loco/g1_loco_api.hpp>
#include <unitree/robot/g1/loco/g1_loco_client.hpp>
#include <iostream>
#include <fstream>
#include <unistd.h>

struct Vel
{
  float x;
  float y;
  float r;
  Vel() : x(0.0f), y(0.0f), r(0.0f) {}
  Vel(float a, float b, float c) : x(a), y(b), r(c) {}
};
int main(int argc, char * argv[])
{

  unitree::robot::ChannelFactory::Instance()->Init(0,"eth0");
  unitree::robot::g1::LocoClient client;
  client.Init();
  client.SetTimeout(10.f);

  std::ifstream velpipe("/tmp/vel_fifo", std::ios::binary);
  Vel value;
  while (true) {
      //std::cout<<"move"<<std::endl;
      //client.Move(0.2, 0.0, 0.0);
      
      velpipe.read(reinterpret_cast<char*>(&value), sizeof(Vel));
      if (velpipe.gcount() == sizeof(Vel)) {
            
            if(value.x==0.0 && value.y==0)
            {
                if(value.r>0.0 && value.r<0.3)
                {
                    value.r = 0.3;
                }
                else
                {
                    if(value.r<0.0 && value.r>-0.3)
                    {
                        value.r = -0.3;
                    }
                }
                
            }
            else
            {
                if(value.r<0.1 && value.r>0.0)
                {
                    value.r = 0.1;
                }
                else
                {
                    if(value.r>-0.1 && value.r<0.0)
                    {
                        value.r = -0.1;
                    }
                }
                
            }

	    if(value.r<-0.3 || value.r>0.3)
	    {
		    if(value.x>0.22)
		    {
		    value.x = 0.22;
		    }
	    }
            std::cout << "Read: " << value.x <<","<<value.y<<","<<value.r << std::endl;
             client.Move(value.x, value.y, value.r);
            //client.Damp();
      } else {
          //std::cout<<"no data"<<std::endl;
          usleep(10000);  // 短暂休眠
      }
      
  }
  
  return 0;
}
