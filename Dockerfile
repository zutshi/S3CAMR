FROM ubuntu:bionic

#RUN useradd -m s3camr
WORKDIR /home

RUN apt-get -y update && 	\
        DEBIAN_FRONTEND=noninteractive apt-get install -y   \
	git-core		\
	vim			\
	make			\
	build-essential		\
	libssl-dev		\
	zlib1g-dev		\
	libbz2-dev		\
	libreadline-dev		\
	libsqlite3-dev		\
	wget			\
	curl			\
	llvm			\
	libncurses5-dev		\
	xz-utils		\
	tk-dev			\
	libxml2-dev		\
	libxmlsec1-dev		\
	libffi-dev		\
        python3.7
        #graphviz                                            \

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py &&\
	 python3.7 get-pip.py









ENV EXT /home/external
WORKDIR /home/external

#Install Z3
RUN wget https://github.com/Z3Prover/z3/archive/z3-4.8.4.tar.gz &&						\
	tar xf ./z3-4.8.4.tar.gz &&										\
	cd $EXT/z3-z3-4.8.4 &&									\
	mkdir $EXT/z3-z3-4.8.4/installation &&							\
	python3.7 scripts/mk_make.py &&			\
	cd build &&												\
	make &&													\
	make install &&												\
	cp -R $EXT/z3-z3-4.8.4/build/python/z3 $EXT/z3-z3-4.8.4/installation/

ENV LD_LIBRARY_PATH $EXT'/z3-z3-4.8.4/installation/lib'
ENV PYTHONPATH $EXT'/z3-z3-4.8.4/installation'

#Install SAL and yices2
RUN wget http://sal.csl.sri.com/opendownload/sal-3.3-bin-x86_64-unknown-linux-gnu-no-yices.tar.gz &&		\
	tar -xf sal-3.3-bin-x86_64-unknown-linux-gnu-no-yices.tar.gz &&						\
	wget http://yices.csl.sri.com/releases/2.6.1/yices-2.6.1-x86_64-pc-linux-gnu-static-gmp.tar.gz &&	\
        tar xf ./yices-2.6.1-x86_64-pc-linux-gnu-static-gmp.tar.gz &&                                           \
	cd sal-3.3 &&												\
	./install.sh &&												\
	echo '(sal/set-yices2-command! "$EXT/yices-2.6.1/bin/yices")' >> ~/.salrc

ENV SAL_PATH "$EXT/sal-3.3"

#Install glpk
RUN wget https://ftp.gnu.org/gnu/glpk/glpk-4.65.tar.gz &&							\
	tar -xf ./glpk-4.65.tar.gz && \
        cd $EXT/glpk-4.65 && \
        mkdir $EXT/glpk-4.65/installation &&\
        ./configure --with-gmp && \
        make CFLAGS=-O2 -j8 && \
        make install

#USER s3camr
ENV HOME /home/S3CAMR
WORKDIR $HOME

RUN git clone https://github.com/zutshi/S3CAMR.git .

RUN git checkout develop &&					\
	git clone https://github.com/zutshi/pyutils.git

ENV PYTHONPATH $PYTHONPATH:/home/S3CAMR/pyutils

#USER root
RUN pip3 install -r requirements.txt

#USER s3camr


# TODO: FORGOTTEN
#USER root
#RUN apt install sudo
#RUN echo 'root:dock' | chpasswd
#USER s3camr
RUN cd $HOME && git pull
RUN cd $HOME/pyutils && git pull


#USER root
WORKDIR $HOME

# python3.7 ./scamr.py -f ../examples/vdp/vanDerPol.tst -cn --refine model-dft --seed 0 --max-model-error 10 --prop-check --bmc-engine sal --opt-engine scipy --incl-error -pmp --plots x0-x1


# # FOr installing graph-tool
# RUN apt-get update && apt-get install --yes --no-install-recommends \
# 	gnupg2

# RUN echo "deb http://downloads.skewed.de/apt/bionic bionic universe" >> /etc/apt/sources.list && \
# 	echo "deb-src http://downloads.skewed.de/apt/bionic bionic universe" >> /etc/apt/sources.list

# RUN apt-key adv --keyserver pgp.skewed.de --recv-key 612DEFB798507F25

# RUN apt-get update && apt-get install --yes --no-install-recommends \
# 	python3-graph-tool=2.27-4

