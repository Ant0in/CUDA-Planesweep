# Makefile for the Planesweep CUDA project
# it is a small sample that pipelines the cmake workflow.
# dont use that if you are on windows, im just the wsl guy

BUILD_DIR := build

all:
	@mkdir -p $(BUILD_DIR)
	@cmake -S . -B $(BUILD_DIR)
	@$(MAKE) -C $(BUILD_DIR) --no-print-directory
	@cp $(BUILD_DIR)/bin/PlaneSweep . 

clean:
	@rm -rf $(BUILD_DIR) PlaneSweep

.PHONY: all clean