variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-east1"
}

variable "zone" {
  description = "GCP Zone"
  type        = string
  default     = "us-east1-b"
}

variable "machine_type" {
  description = "GCP Machine Type"
  type        = string
  default     = "n2-standard-8"
}

variable "gpu_count" {
  description = "Number of GPUs"
  type        = number
  default     = 0
}

variable "gpu_type" {
  description = "Type of GPU"
  type        = string
  default     = "nvidia-tesla-t4"
}

variable "model_id" {
  description = "Model ID to serve (not used in CPU mode)"
  type        = string
  default     = "llmat/Qwen3-8B-NVFP4"
}

variable "hf_token" {
  description = "Hugging Face Read Token (dummy for CPU mode)"
  type        = string
  sensitive   = true
  default     = "dummy"
}
