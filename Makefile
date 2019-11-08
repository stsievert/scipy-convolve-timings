
# | Name | Single core | Multi-core |
# | "MacBook Pro (15-inch Retina Mid 2014)" | 861 | 3150 | (loaner from Michelle)
# | "MacBook Pro (13-inch Retina Early 2015) " | 1065 | 4005 | (new machine coming soon)
# | "MacBook Pro (13-inch Retina Early 2015) " | 793 | 1657 | (higest)
# | "MacBook Pro (13-inch Retina Early 2015) " | 765 | 1621 | (middle)

setup-docker:
	cd ../scipy/
	docker run -it --rm -v $PWD/:/home/scipy scipy/scipy-dev /bin/bash

build-scipy-on-docker:
	cd /home/scipy
	pip3.7 install numpy cython pytest pybind11
	python3.7 setup.py build_ext --inplace  # takes a while
	python3.7 runtests.py -v  # takes a while
	python3.7 setup.py develop  # takes less time than build_ext
	pip3.7 install scikit-learn pandas pyarrow fastparquet

run-timings:
	python3.7 Generate-Train-Data-1d.py
	python3.7 Generate-Train-Data-2d.py
	python3.7 Generate-Test-Data-2d.py
	python3.7 Generate-Test-Data-1d.py
