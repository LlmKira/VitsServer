#!/bin/bash

# Check if the VitsServer directory already exists
if [ -d "VitsServer" ]; then
    echo "VitsServer directory already exists, updating..."
    cd VitsServer
    git pull --exclude=.env --exclude=model
    exit
else
    # Clone the repository
    git clone https://github.com/LlmKira/VitsServer.git
    cd VitsServer
fi

# Update system packages
sudo apt-get update &&
sudo apt-get install -y build-essential libsndfile1 vim gcc g++ cmake

# Install Python dependencies
sudo apt install python3-pip
pip3 install pipenv

# Install dependency packages
pipenv install

# Activate the virtual environment
pipenv shell

# Set up the configuration file
touch .env
echo "VITS_SERVER_HOST=0.0.0.0" > .env
echo "VITS_SERVER_PORT=9557" >> .env
echo "VITS_SERVER_RELOAD=false" >> .env

# Start the server using PM2
sudo apt install npm
npm install pm2 -g
pm2 start pm2.json --name vits_server --watch

# Save the PM2 process list so it will be started at boot
sudo pm2 save

# Exit the virtual environment
exit
