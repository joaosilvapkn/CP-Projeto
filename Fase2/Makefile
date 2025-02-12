CPP = g++ -Wall -Ofast -fopenmp -fno-omit-frame-pointer
CPP2 = g++ -Ofast
SRCS = src/main.cpp src/fluid_solver.cpp src/EventManager.cpp
TARGET = fluid_sim

all: phase2 phase1

phase1:
	$(CPP2) $(SRCS) -o fluid_sim_seq

phase2:
	$(CPP) $(SRCS) -o $(TARGET)

clean:
	@echo Cleaning up...
	@rm -f $(TARGET) fluid_sim_seq
	@echo Done.

runseq:
	./fluid_sim_seq

runpar:
	export OMP_NUM_THREADS=40 && ./$(TARGET)
