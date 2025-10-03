all: cuda_vector_add cuda_vector_add_err

cuda_vector_add: cuda_vector_add.cu
	nvcc -o cuda_vector_add ./cuda_vector_add.cu 

cuda_vector_add_err: cuda_vector_add_err.cu
	nvcc -o cuda_vector_add_err ./cuda_vector_add_err.cu 

clean:
	rm -f cuda_vector_add cuda_vector_add_err
