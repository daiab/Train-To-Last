# using google tracing
http://mxnet.io/how_to/perf.html#profiler

# set prefetch_buffer
http://mxnet.io/zh/architecture/note_data_loading.html?highlight=prefetch_buffer

# python 接口层是怎样与cpp代码衔接起来的?
mxnet python 里提供的大部分接口方法都不是在python中直接定义, 而是通过类似于注入的方式将cpp代码中定
义好的方法复制到python对象里. 那么这个过程是什么时候, 在哪里完成的呢?在io.py里有_init_io_module()方法,
它将mxnet.so中的data iterator都取出来, 然后作为属性赋给对应的python对象: mx.io. 其他一些需要调用
底层cpp方法的接口, 应该也是通过这种办法与cpp代码衔接起来的, 如symbol.py里对应的也有_init_symbol_module()
方法.那么, 这个注入的过程是什么时候发生的呢? 第一次import相应的模块时发生, 通过执行对应的_init_xxx_module()方法.
