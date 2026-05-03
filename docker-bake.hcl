variable "FIDO_IMAGE" {
  default = "fido:local"
}

variable "FIDO_TEST_IMAGE" {
  default = "fido-test:local"
}

variable "FIDO_ROCQ_REPL_IMAGE" {
  default = "fido-rocq-repl:local"
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

target "rocq-repl" {
  context = "."
  dockerfile = "models/Dockerfile"
  target = "rocq-repl"
  tags = [FIDO_ROCQ_REPL_IMAGE]
  output = ["type=docker"]
  args = {
    ROCQ_IMAGE = "rocq_image"
  }
  contexts = {
    rocq_image = "target:rocq-image"
  }
}

target "rocq-image" {
  context = "rocq-python-extraction"
  dockerfile = "Dockerfile"
}

target "make-rocq" {
  context = "."
  dockerfile = "models/Dockerfile"
  target = "export"
  output = ["type=local,dest=."]
  args = {
    ROCQ_IMAGE = "rocq_image"
  }
  contexts = {
    rocq_image = "target:rocq-image"
    rocq_models_cache = ".cache/rocq-models/context"
  }
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

// The "test" Dockerfile stage is no longer in the ci bake group: buildx
// bake offers no per-target memory cap so a leaky test could grow
// unbounded inside the buildkit cgroup and soft-lock the host (#1248).
// ``./fido ci`` invokes ``./fido tests`` after bake — the tests run via
// the host-side capped runner (``run_container --memory=4g``).
group "ci" {
  targets = ["format", "lint", "typecheck", "generated-typecheck", "fido", "rocq-repl"]
}
