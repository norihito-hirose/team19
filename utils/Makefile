DEP := $(shell command -v dep 2> /dev/null)

app_name := "create_repo_list"
bin_path := "../bin"
build : dep
	go build -o $(bin_path)/$(app_name) main.go
dep :
ifndef DEP
	go get -u github.com/golang/dep/cmd/dep
endif
	$(shell echo "installed dep tool." >&2)
	dep ensure

clean :
	rm $(bin_path)/$(app_name)
