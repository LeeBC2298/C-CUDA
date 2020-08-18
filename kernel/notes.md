# 编写自己的.cu程序
kernel.h和kernel.cu已上传，供参考

## kernel.h
类似C++的.h文件，首先include必要的库，比如cuda_runtime.h

然后声明一个wrapper函数，这个wrapper函数完成一些调用kernel前的预处理，从而方便其他项目的调用，可以根据自己的需要设计wrapper函数的参数和返回值

## kernel.cu
主要的代码都在kernel.cu中实现
### kernel
通过__global__ viod kernel()声明需要cuda加速的函数

注意__global__是必须的，这会告诉编译器哪个函数需要由GPU处理

在kernel中，通过blockIdx、blockDim、threadIdx就可以完成寻址，例如对于二维grid和block：

&nbsp;int y = blockIdx.y * blockDim.y + threadIdx.y;
&nbsp;int x = blockIdx.x * blockDim.x + threadIdx.x;

### wrapper
wrapper函数可以根据自己的需要设计参数和返回值，比如做单张图片处理那就输入一个CV::Mat

在调用kernel之前，需要先把CPU内存中的数据搬到显存里去，所以需要：

&nbsp;unsigned char* src_dev声明一个gpu type的源数据指针  
&nbsp;cudaMalloc为src_dev开辟一块显存空间  
&nbsp;cudaMemcpy把内存数据copy到显存中

然后就可以指定grid和block来调用kernel函数了：

&nbsp;kernel<<<grid, block>>>()

grid、block都是cuda的一种数据结构“dim3”，在我的实验中，block表示的是一个块中有多少个线程，grid则表示一共有多少个block。在实验中，block取（16， 16， 1）的效果是最好的，也就是一个block中有256个线程。

调用kernel后，记得把显存中的数据通过cudaMemcpy拷贝回内存再返回，然后用cudaFree释放显存空间。
