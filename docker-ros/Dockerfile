FROM ubuntu:18.04

LABEL Description="ROS melodic compiled for python3 in Ubuntu 18.04" Version="1.0"

#
# ========================== General Ubuntu Settings ==========================
#

RUN printf '\n\n Applying Ubuntu Settings.. \n\n'

#
# Define script parameters
#
ARG shell=/bin/bash

# Replace pathToAppDir with the desired container's path 
# to the root of the ros-based application, named <AppID>. 
# As an example, we set <AppID> = SimulationManager
ARG pathToAppDir="/opt/SimulationManager"

# Path to the catkin workspace of the ROS-based application.
ARG ros_ws="${pathToAppDir}/SimulationManager/ROS/catkin_build_ws"

# Use /bin/bash instead of /bin/sh
RUN mv /bin/sh /bin/sh-old && \
ln -s /bin/bash /bin/sh

# Set timezone based on your location
ENV TZ=Europe/Rome
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# install apt-utils, git, python3, pip3
RUN apt-get update -y && apt-get install -y \ 
    apt-utils \
    python3.6 python-pip python3-pip \
    git \
    lsb-release mesa-utils \
    software-properties-common locales x11-apps \
    gedit gedit-plugins nano \
    screen tree \
    sudo ssh synaptic \
    wget curl unzip htop \
    gdb valgrind \
    libcanberra-gtk* \
    xsltproc \
    libgtest-dev \
    iputils-ping iproute2 \
&& rm -rf /var/lib/apt/lists/* \
&& apt-get clean


#
# ========================== ROS Setup ==========================
#

RUN printf '\n\n Installing ROS.. \n\n'

# Install random stuff that is missing in the ros installation
RUN pip3 install -U netifaces gnupg empy rospkg numpy
RUN apt-get update -y && apt-get install -y \
    sip-dev \
    pyqt5-dev \
    python3-sip-dev \
    pyqt5-dev-tools \
    python-catkin-pkg \
&& rm -rf /var/lib/apt/lists/* \
&& apt-get clean

# Install python3 equivalent packages for ros
RUN pip3 install -U rosdep rosinstall-generator wstool queuelib nose pyros_setup pyyaml catkin-tools rostful opencv-contrib-python defusedxml
#The following python 2 pkg is still required for catkin_make to run at the catkin_ws
RUN pip install -U catkin_pkg

# Install packages to build ros using python 3
RUN apt-get update -y && apt-get install -y \ 
    python3-rosinstall build-essential
RUN pip3 uninstall vcstools -y

# Use ONLY this specific version of vcstool
RUN git clone -b mock_server_tar_test https://github.com/tkruse/vcstools.git \
&& pip3 install ./vcstools/

# Install rosdep to deal with dependencies of further ROS packages
RUN rm -rf /etc/ros/rosdep/sources.list.d/* \
&& rosdep init \
&& rosdep update

RUN echo "PART two of dockerfile installation"

# Link python 3 dist to site pckgs
RUN ln -s /usr/local/lib/python3.6/dist-packages /usr/local/lib/python3.6/site-packages

# Get and compile all melodic pckgs
RUN mkdir ~/catkin_ws && cd ~/catkin_ws 
RUN rosinstall_generator desktop --rosdistro melodic --deps --tar > melodic-desktop.rosinstall
RUN wstool init -j8 src melodic-desktop.rosinstall 
# questa app installa anche "python-catkin-pkg-modules" che ha problemi con `rosdep install`
RUN apt-get install -y python-catkin-pkg
RUN rosdep install --from-paths src --ignore-src --rosdistro melodic -y --skip-keys="python-catkin-pkg-modules python-rosdep-modules"
RUN export PYTHONPATH=/usr/local/lib/python3.6/dist-packages
RUN ./src/catkin/bin/catkin_make_isolated --install --install-space /opt/ros/melodic -DPYTHON_EXECUTABLE=/usr/bin/python3 -DCMAKE_BUILD_TYPE=Release

# you MUST source the ros env or it will not work -> copy the env to bashrc
RUN echo "source /opt/ros/melodic/setup.sh" >> ~/.bashrc

#
# ========================== Setup the ROS-based Application (<AppID> = SimulationManager) ==========================
#

RUN printf '\n\n Installing SimulationManager.. \n\n'

#Create the <AppID>'s catkin workspace folder for ROS source files
RUN mkdir -p ${ros_ws}/src

# Copy the contents of the host's existing app into the container's ROS-based app.
# Set and uncomment the following variables for the desired ROS-based application of yours.
#COPY /SimulationManager ${pathToAppDir}
#RUN chmod -R 776 ${pathToAppDir}

# try compiling the ${ros_ws} for the usage of <AppID>
WORKDIR ${ros_ws}

#Try to issue a catkin_make in the current directory
RUN /bin/bash -c ". /opt/ros/melodic/setup.sh \
&& rm -rf /devel /build /logs /src/CMakeLists.txt \
&& cd src \
&& catkin_init_workspace ${ros_ws}/src\
&& cd .. \
&& catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so \
&& catkin config --install \
&& catkin_make"

# you MUST source the ros env or it will not work -> copy the env to bashrc
RUN echo "source ${ros_ws}/devel/setup.bash" >> ~/.bashrc
RUN /bin/bash -c ". ${ros_ws}/devel/setup.bash"

#
# ========================== Tensorflow, keras, scipy ==========================
#

ARG catkin_ws="/root/catkin_ws"

# Install trac-ik
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN sudo apt-get update
# this will fail
RUN apt-get install -y ros-melodic-moveit; exit 0
RUN apt --fix-broken install -y
RUN apt-get install -y ros-melodic-moveit swig libnlopt-dev ros-melodic-joint-state-publisher-gui

WORKDIR ${catkin_ws}
RUN mkdir src
WORKDIR ${catkin_ws}/src
COPY trac_ik trac_ik
COPY trac_ik_examples trac_ik_examples
COPY trac_ik_kinematics_plugin trac_ik_kinematics_plugin
COPY trac_ik_lib trac_ik_lib
COPY trac_ik_python trac_ik_python

# Install tensorflow
RUN pip3 install numpy scipy matplotlib ipython jupyter pandas sympy nose gym PyInquirer wandb
RUN pip3 install --upgrade tensorflow
RUN pip3 install --upgrade keras

# Install keras-rl
WORKDIR /app
COPY keras-rl keras-rl
WORKDIR /app/keras-rl 
RUN python3 setup.py install

# Install kinpy (changed setup.py, added MANIFEST.in)
WORKDIR /app
COPY kinpy kinpy
WORKDIR /app/kinpy
RUN python3 setup.py install

#
# ========================== Final environment configs ==========================
#

# RUN 
RUN /bin/bash -c "source /opt/ros/melodic/setup.bash && export ROS_PYTHON_VERSION=3 && cd ~/catkin_ws && catkin_make"

# install GLX-Gears and the GL Vendor-Neutral Dispatch library
RUN apt-get update && apt-get install -y \
    libglvnd0 \
    mesa-utils && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install PyInquirer

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# Instructions
RUN echo "echo 'Follow this steps'" >>  ~/.bashrc \
&& echo "echo '- On HOST:'" >>  ~/.bashrc \
&& echo "echo '1. xhost +local'" >>  ~/.bashrc \
&& echo "echo '2. sudo nvidia-docker run -it --env=\"DISPLAY\" --volume=\"/tmp/.X11-unix:/tmp/.X11-unix:rw\" --volume=\"/home/incud/Documents/reinforcement_learning/docker-ros-volume:/root/docker-ros-volume\" ros-for-exam:1.2 bash'" >> ~/.bashrc \
&& echo "echo '- On DOCKER:'" >>  ~/.bashrc \
&& echo "echo '1. ROS_PYTHON_VERSION=3'" >>  ~/.bashrc \
&& echo "echo '2. cp -a /root/docker-ros-volume/snake /root/catkin_ws/src/ && cd ~/catkin_ws && catkin_make'" >>  ~/.bashrc \
&& echo "echo '3. source ~/catkin_ws/devel/setup.bash'" >>  ~/.bashrc \
&& echo "echo '4. roslaunch snake snake.launch'" >>  ~/.bashrc

# Set the image to start opening a new bash terminal
WORKDIR /root
ENTRYPOINT ["/bin/bash", "-c"]
