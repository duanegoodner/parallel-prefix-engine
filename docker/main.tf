provider "aws" {
  region = "us-west-2"
}

resource "aws_instance" "gpu_instance" {
  ami           = "ami-0e020752b7ce0a61f"  # Ubuntu 22.04 in us-west-2 with GPU support
  instance_type = "g4dn.xlarge"
  key_name      = "your-ssh-key-name"  # Replace with your actual key name

  root_block_device {
    volume_size = 100
  }

  user_data = <<-EOF
              #!/bin/bash
              apt-get update
              apt-get install -y docker.io
              systemctl start docker
              systemctl enable docker
              
              # Install NVIDIA drivers
              apt-get install -y build-essential dkms
              ubuntu-drivers devices
              ubuntu-drivers autoinstall
              reboot
              EOF

  tags = {
    Name = "cuda-docker-node"
  }
}

output "public_ip" {
  value = aws_instance.gpu_instance.public_ip
} 
