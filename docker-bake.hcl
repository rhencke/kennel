variable "FIDO_IMAGE" {
  default = "fido:local"
}

variable "FIDO_TEST_IMAGE" {
  default = "fido-test:local"
}

variable "FIDO_UID" {
  default = "1000"
}

variable "FIDO_GID" {
  default = "1000"
}

variable "FIDO_USER" {
  default = "fido"
}

variable "FIDO_HOME" {
  default = "/home/fido"
}

target "fido" {
  context = "."
  dockerfile = "models/Dockerfile"
  target = "fido"
  tags = [FIDO_IMAGE]
  output = ["type=docker"]
  args = {
    FIDO_UID = FIDO_UID
    FIDO_GID = FIDO_GID
    FIDO_USER = FIDO_USER
    FIDO_HOME = FIDO_HOME
  }
}

target "fido-test" {
  context = "."
  dockerfile = "models/Dockerfile"
  target = "fido-test"
  tags = [FIDO_TEST_IMAGE]
  output = ["type=docker"]
  args = {
    FIDO_UID = FIDO_UID
    FIDO_GID = FIDO_GID
    FIDO_USER = FIDO_USER
    FIDO_HOME = FIDO_HOME
  }
}

target "rocq-image" {
  context = "rocq-python-extraction"
  dockerfile = "Dockerfile"
}

target "format" {
  context = "."
  dockerfile = "models/Dockerfile"
  target = "format"
  args = {
    ROCQ_IMAGE = "rocq_image"
  }
  contexts = {
    rocq_image = "target:rocq-image"
    rocq_models_cache = ".cache/rocq-models/context"
  }
}

target "lint" {
  context = "."
  dockerfile = "models/Dockerfile"
  target = "lint"
  args = {
    ROCQ_IMAGE = "rocq_image"
  }
  contexts = {
    rocq_image = "target:rocq-image"
    rocq_models_cache = ".cache/rocq-models/context"
  }
}

target "typecheck" {
  context = "."
  dockerfile = "models/Dockerfile"
  target = "typecheck"
  args = {
    ROCQ_IMAGE = "rocq_image"
  }
  contexts = {
    rocq_image = "target:rocq-image"
    rocq_models_cache = ".cache/rocq-models/context"
  }
}

target "generated-typecheck" {
  context = "."
  dockerfile = "models/Dockerfile"
  target = "generated-typecheck"
  args = {
    ROCQ_IMAGE = "rocq_image"
  }
  contexts = {
    rocq_image = "target:rocq-image"
    rocq_models_cache = ".cache/rocq-models/context"
  }
}

target "test" {
  context = "."
  dockerfile = "models/Dockerfile"
  target = "test"
  args = {
    ROCQ_IMAGE = "rocq_image"
  }
  contexts = {
    rocq_image = "target:rocq-image"
    rocq_models_cache = ".cache/rocq-models/context"
  }
}

group "ci" {
  targets = ["format", "lint", "typecheck", "generated-typecheck", "test"]
}

group "warm" {
  targets = ["format", "lint", "typecheck", "generated-typecheck", "test", "fido"]
}
