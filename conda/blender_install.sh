cd ..
wget https://builder.blender.org/download/daily/archive/blender-2.83.20-stable+v283.a56e2faeb7a9-linux.x86_64-release.tar.xz
sudo tar -xvf blender-2.83.20* --strip-components=1 -C /bin
rm -rf blender-2.83.20*
rm -rf blender-2.83.20*
sudo rm -rf /bin/2.83/python
sudo ln -s /home/$USER/anaconda3/envs/ns_ap /bin/2.83/python
