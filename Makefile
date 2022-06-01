CC:=g++
TFPREFIX:=/home/xwy/miniconda3/envs/pack1/lib/python3.9/site-packages/tensorflow/
INC_DIR:=  -I$(TFPREFIX)/include
LINK_DIR:= -L$(TFPREFIX)/lib
LINK_FLAGS:=-shared 
CC_FLAGS:=-std=c++14 -g -w -fPIC -D_GLIBCXX_USE_CXX11_ABI=0
# LINK_LIB:=-ltensorflow_framework -Wl,-rpath=$(TFPREFIX)/lib
LINK_LIB:= -Wl,-rpath=$(TFPREFIX)/lib
libtimer.so: timer.o $(TFPREFIX)/libtensorflow_framework.so.2
		$(CC) $(CC_FLAGS) $(LINK_FLAGS) $(LINK_DIR) $(LINK_LIB) $^ -o $@
timer.o: cc/timer_kernels.cc
		$(CC) $(CC_FLAGS) $(INC_DIR) -c $^ -o $@
clean:
		rm *.o *.so
