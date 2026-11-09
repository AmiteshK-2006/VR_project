#!/bin/bash
cd ~/snap/snapd-desktop-integration/253/Desktop/vr_project || exit
git pull origin main
git add .
git commit -m "Auto commit: $(date '+%Y-%m-%d %H:%M:%S')"
git push origin main
