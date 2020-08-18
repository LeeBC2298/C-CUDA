# 利用CMakeList.txt将原项目和cuda程序（.cu）文件混合编译
.cu文件是cuda加速程序的后缀，其语法大体和C++一致，但是编译.cu文件需要使用nvcc命令，而不能使用gcc等。

直接在原项目的CMakeList.txt的sources中加入.cu文件也可以完成编译，但是出于工程应用的考虑，我选择将它单独编译成动态库，然后给主项目链接。这样可以尽量减少对原项目的修改。

## 文件结构
project  
|-main.cpp  
|-CMakeList.txt  
|-gpu_acceleration  
&emsp;|-kernel.cu  
&emsp;|-kernel.h  
&emsp;|-CMakeList.txt

main.cpp是原项目主要文件，gpu_acceleration是包含cuda加速程序的文件夹。

## CMakeList
gpu_acceleration的CMakeList.txt已经上传，可以直接参考。

原项目的CMakeList只要添加链接操作即可：

&nbsp;&nbsp;SET(EXTRA_LIBS ${EXTRA_LIBS} gpu_acceleration)  

&nbsp;&nbsp;TARGET_LINK_LIBRARIES(${PROG_NAME}  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;...(项目原来链接的库)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;${EXTRA_LIBS}  
&nbsp;&nbsp;)
