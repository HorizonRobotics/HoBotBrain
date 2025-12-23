#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"

// #include <unitree/robot/g1/loco/g1_loco_api.hpp>
// #include <unitree/robot/g1/loco/g1_loco_client.hpp>
#include <queue>
#include <mutex>
#include <iostream>
#include <fstream>
#include <unistd.h>
std::ofstream velpipe("/tmp/vel_fifo", std::ios::binary);
struct Vel
{
  float x;
  float y;
  float r;
  Vel() : x(0.0f), y(0.0f), r(0.0f) {}
  Vel(float a, float b, float c) : x(a), y(b), r(c) {}
};
class VelocitySubscriber : public rclcpp::Node
{
public:
  VelocitySubscriber()
  : Node("velocity_subscriber")
  {
     RCLCPP_INFO(this->get_logger(), "订阅速度话题启动，等待数据...");
     
    // 创建订阅者，订阅"/cmd_vel"话题，消息类型为geometry_msgs::msg::Twist
   // subscription_ = this->create_subscription<geometry_msgs::msg::Twist>("/cmd_vel", 10,std::bind(&VelocitySubscriber::topic_callback, this, std::placeholders::_1, x));
    subscription_ = create_subscription<geometry_msgs::msg::Twist>(
      "/cmd_vel",
      10,
      [this](geometry_msgs::msg::Twist::SharedPtr msg) {
          // 提取线速度和角速度
            double linear_x = msg->linear.x;
            double linear_y = msg->linear.y;
            double linear_z = msg->linear.z;
    
            double angular_x = msg->angular.x;
            double angular_y = msg->angular.y;
            double angular_z = msg->angular.z;
 
            // 打印速度信息
            RCLCPP_INFO(this->get_logger(), "线速度: x=%.2f, y=%.2f, z=%.2f m/s", 
                linear_x, linear_y, linear_z);
            RCLCPP_INFO(this->get_logger(), "角速度: x=%.2f, y=%.2f, z=%.2f rad/s",
                angular_x, angular_y, angular_z);
            Vel value(linear_x, linear_y, angular_z);    
            velpipe.write(reinterpret_cast<char*>(&value), sizeof(Vel));
            velpipe.flush();  // 确保数据立即写入
            // client_.Move(linear_x, linear_y, angular_z);
            //client_.Damp();
       }
    );
  }
  
  // void setClient(unitree::robot::g1::LocoClient client)
  // {
  //     client_ = client;
  // }

private:
  
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr subscription_;
  
  //std::mutex m_buf;
  //std::queue<geometry_msgs::msg::Twist> velbuf;
  // unitree::robot::g1::LocoClient client_;
};

int main(int argc, char * argv[])
{
  //setenv("ROS_DOMAIN_ID", "1", 1);

  // unitree::robot::ChannelFactory::Instance()->Init(0,"eth0");
  // unitree::robot::g1::LocoClient client;
  // client.Init();
  // client.SetTimeout(10.f);
  
  rclcpp::init(argc, argv);
  auto node = std::make_shared<VelocitySubscriber>();
  // node->setClient(client);

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
