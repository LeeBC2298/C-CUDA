# 编写自己的.cu程序
kernel.h和kernel.cu已上传，是我自己写的一个图像预处理的样例，预处理的内容包括resize（最邻近差值）、BGR2RGB和图像归一化，供参考

## kernel.h
类似C++的.h文件，首先include必要的库，比如cuda_runtime.h

然后声明一个wrapper函数，这个wrapper函数完成一些调用kernel前的预处理，从而方便其他项目的调用，可以根据自己的需要设计wrapper函数的参数和返回值

## kernel.cu
主要的代码都在kernel.cu中实现
### kernel
通过__global__ viod kernel()声明需要cuda加速的函数

注意__global__是必须的，这会告诉编译器哪个函数需要由GPU处理

在kernel中，通过blockIdx、blockDim、threadIdx就可以完成寻址，例如对于二维grid和block：

&nbsp;&nbsp;&nbsp;int y = blockIdx.y * blockDim.y + threadIdx.y;
&nbsp;&nbsp;&nbsp;int x = blockIdx.x * blockDim.x + threadIdx.x;

### wrapper
wrapper函数可以根据自己的需要设计参数和返回值，比如做单张图片处理那就输入一个CV::Mat

在调用kernel之前，需要先把CPU内存中的数据搬到显存里去，所以需要：

&nbsp;&nbsp;&nbsp;unsigned char* src_dev声明一个gpu type的源数据指针  
&nbsp;&nbsp;&nbsp;cudaMalloc为src_dev开辟一块显存空间  
&nbsp;&nbsp;&nbsp;cudaMemcpy把内存数据copy到显存中

然后就可以指定grid和block来调用kernel函数了：

&nbsp;&nbsp;&nbsp;kernel<<<grid, block>>>()

grid、block都是cuda的一种数据结构“dim3”，在我的实验中，block表示的是一个块中有多少个线程，grid则表示一共有多少个block。在实验中，block取（16， 16， 1）的效果是最好的，也就是一个block中有256个线程。

调用kernel后，记得把显存中的数据通过cudaMemcpy拷贝回内存再返回，然后用cudaFree释放显存空间。

## 运行效率
### 优化cudaMemcpy速度
实际使用时候发现并没有比CPU直接处理快多少。。甚至反而fps有所降低，经过分析发现是cudaMemcpy耗费了太多时间（一次cpy大约是实际gpu运算的10倍）

查阅资料后，了解到可以通过申请pinned内存来提高内存到显存的copy效率：

&nbsp;&nbsp;&nbsp;cudaHostAlloc((void**)&data, batchsize*size*sizeof(float), cudaHostAllocDefault);

注意这里申请的是一块内存而非显存，也就是所它相当于C++的malloc函数而非CUDA的cudaMalloc，它与malloc的区别是，这样申请的内存是不会被置换的，从而可以省去很多检查，加快cudaMemcpu操作

最后，不要忘记用cudaFreeHost把内存释放掉

实际实验后发现这个操作的确让memcpy提速1倍左右，和网上很多教程给出来的效果基本一样，然而在我的项目里，把cv::Mat的data复制到这块内存又会产生多余开销。。。

果然前人留下的框架导致我的努力并没有什么效果。。。必须修改现有框架，把接收的数据从cv::Mat改成pinned内存块才能通过这种方式获得性能提升

### 减少cudaMemcpy次数
考虑batch操作，不再一张一张图处理，待续
