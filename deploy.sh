#!/bin/bash

# Build the project
./build.sh

# Create deploy directory
rm -rf deploy
mkdir deploy

# Copy necessary files
cp index.html deploy/
cp -r pkg deploy/
cp README.md deploy/

echo "Deploy directory created with all necessary files"
echo "Ready for GitHub Pages deployment"