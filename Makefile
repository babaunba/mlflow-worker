# after generation, go to *_pb2_grpc.py and change to `from .`
generate:
	rm -rf gen/
	git clone https://github.com/babaunba/proto.git
	
	mkdir -p gen/proto
	python -m grpc_tools.protoc \
		-I=./proto \
		-I=vendor.proto \
		--python_out=./gen/proto \
		--pyi_out=./gen/proto \
		$$(find ./proto/ -name '*.proto')
	
	rm -rf proto/

vendor:
	git clone --filter=blob:none --sparse https://github.com/googleapis/googleapis.git
	cd googleapis && git sparse-checkout set google/api
	
	mv googleapis/google .
	mkdir vendor.proto || true
	mv google vendor.proto
	rm -rf googleapis
