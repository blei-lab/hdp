.PHONY: make_all

make_all:
	$(MAKE) -C hdp-faster
	$(MAKE) -C hdp

clean:
	$(MAKE) clean -C hdp-faster
	$(MAKE) clean -C hdp
