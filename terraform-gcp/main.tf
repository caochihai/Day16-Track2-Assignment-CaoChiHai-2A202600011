provider "google" {
  project = var.project_id
  region  = var.region
}

# 1. Network
resource "google_compute_network" "vpc" {
  name                    = "ai-lab-vpc"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "subnet" {
  name          = "ai-lab-subnet"
  ip_cidr_range = "10.0.1.0/24"
  network       = google_compute_network.vpc.id
  region        = var.region
}

# 2. Firewall
resource "google_compute_firewall" "allow_http" {
  name    = "allow-http-vllm"
  network = google_compute_network.vpc.name

  allow {
    protocol = "tcp"
    ports    = ["8000"]
  }

  source_ranges = ["0.0.0.0/0"]
}

resource "google_compute_firewall" "allow_ssh" {
  name    = "allow-ssh"
  network = google_compute_network.vpc.name

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = ["0.0.0.0/0"]
}

# 3. GPU Instance (G2 with NVIDIA L4)
resource "google_compute_instance" "gpu_node" {
  name         = "ai-cpu-node"
  machine_type = var.machine_type
  zone         = var.zone

  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-11"
      size  = 100
      type  = "pd-ssd"
    }
  }

  # guest_accelerator {
  #   type  = var.gpu_type
  #   count = var.gpu_count
  # }

  network_interface {
    subnetwork = google_compute_subnetwork.subnet.id
    access_config {
      # Public IP for testing
    }
  }

  metadata_startup_script = templatefile("${path.module}/user_data.sh", {
    hf_token = var.hf_token
    model_id = var.model_id
  })

  scheduling {
    on_host_maintenance = "MIGRATE"
    automatic_restart   = true
  }
}

output "public_ip" {
  value = google_compute_instance.gpu_node.network_interface[0].access_config[0].nat_ip
}
