# Deploying MedRAG-lite to AWS EC2 (Free Tier Optimized)

This guide walks you through taking your Dockerized MedRAG-lite application and putting it on the internet using Amazon Web Services (AWS) EC2, absolutely free!

## Step 1: Push your code to GitHub
Before touching AWS, make sure your code (including the new `Dockerfile` and `requirements.txt`) is pushed to a GitHub repository. This makes getting your code onto the cloud server very easy.

## Step 2: Create an AWS Account and Launch an EC2 Instance
1. Go to [aws.amazon.com](https://aws.amazon.com/) and create a free account.
2. Search for **EC2** in the AWS console search bar and click on it.
3. Click **Launch Instance**.
4. **Name:** `medrag-server`
5. **AMI (OS):** Select **Ubuntu** (Ubuntu Server 24.04 or 22.04 LTS).
6. **Instance Type:** Select **t2.micro** or **t3.micro** (Whichever says "Free tier eligible").
7. **Key Pair:** Click **Create new key pair**, name it `medrag-key`, and download it. You will need this to connect to the server!
8. **Network Settings:** Check these boxes under "Allow traffic":
   - [x] Allow SSH traffic from Anywhere
   - [x] Allow HTTP traffic from the internet
   - [x] Allow HTTPS traffic from the internet
9. Click **Launch Instance**.

## Step 3: Connect to your Server
1. Go back to the EC2 Dashboard, click on your running `medrag-server` instance.
2. Copy the **Public IPv4 address**.
3. Open a terminal on your computer, navigate to where you saved your `medrag-key.pem` file, and run:
   ```bash
   # On Mac/Linux, set permissions first:
   chmod 400 medrag-key.pem
   
   # Connect to the server:
   ssh -i "medrag-key.pem" ubuntu@YOUR_PUBLIC_IP_ADDRESS
   ```

## Step 4: The DevOps Secret (Add Swap Space)
Since the free tier only has 1GB of RAM (which isn't enough for AI models), we will create a "Swap File". This tricks the server into using 4GB of your free hard drive space as extra RAM so the app doesn't crash!
Run these commands exactly as written:
```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
# Make it permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

## Step 5: Install Docker
Run these commands to install Docker:
```bash
# Update packages
sudo apt update -y

# Install Docker
sudo apt install docker.io -y

# Allow your user to run Docker without typing 'sudo' every time
sudo usermod -aG docker ubuntu
```
*(You must log out by typing `exit` and log back in for the permission change to take effect).*

## Step 6: Clone Your Code and Run It
1. Clone your repository onto the server:
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
   cd YOUR_REPOSITORY
   ```
2. Build your Docker container (this will take a few minutes as it downloads PyTorch):
   ```bash
   docker build -t medrag-app .
   ```
3. Run your Docker container:
   ```bash
   docker run -d -p 80:5000 medrag-app
   ```
   *Explanation: `-d` runs it in the background. `-p 80:5000` maps port 80 to port 5000.*

## Step 7: Access Your App
Open your web browser and navigate to your EC2 instance's Public IP address:
`http://YOUR_PUBLIC_IP_ADDRESS`

Your MedRAG application is now live on the internet!
