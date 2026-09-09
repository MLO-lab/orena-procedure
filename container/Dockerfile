FROM --platform=linux/amd64 docker.io/pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime AS deps

# The template recommends 2.11.0-cuda12.8. This is the same image line one torch minor
# back, and the swap is deliberate: measured on the same node, same weights, same
# question, torch 2.11.0 generates in 7.8 s where 2.9.1 takes 4.2 s, and holds 4.4 GiB
# more VRAM. 2.9.1 is also exactly what this checkpoint was trained and evaluated with.
#
# Everything the template's recommendation is FOR still holds, which is why this is safe:
# the CUDA version is unchanged at 12.8, so the driver floor is unchanged and one image
# still serves both evaluation GPUs, and this build's kernels (sm_70 through sm_120)
# cover the RTX PRO 6000 natively and the L40S via sm_86 -- a superset of 2.11's. The
# guard below is the template's own and is deliberately left asserting CUDA 12.8, so a
# base-image change that DID move the driver floor would still fail the build.

ENV PYTHONUNBUFFERED=1

# Frame counts vary per question, so the caching allocator strands memory inside
# partly-used blocks; expandable segments recover it. Purely an allocation strategy --
# identical tensors, identical answers. Both spellings are set because the new name
# alone did not take effect.
ENV PYTORCH_ALLOC_CONF=expandable_segments:True \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Every cache torch and its dependencies may write at runtime, pointed at /tmp. The
# platform mounts /tmp as a volume and gives no promise that anything else is writable;
# a library that defaults to $HOME and finds it read-only raises rather than degrades.
# Observed: with flash-linear-attention installed, triton tried to create ~/.triton and
# the run died with "Read-only file system".
ENV TRITON_CACHE_DIR=/tmp/triton \
    TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor \
    XDG_CACHE_HOME=/tmp/cache \
    HF_HOME=/tmp/huggingface \
    MPLCONFIGDIR=/tmp/matplotlib

# The slim -runtime base ships neither the shared libraries opencv-python (a dependency
# of orena-focus) needs at import time, nor a compiler toolchain.
#
# No compiler toolchain is installed, and nothing here needs one -- see the note on
# flash-linear-attention in requirements.txt, which is the dependency that would.
#
# APT::Sandbox::User=root stops apt dropping to the _apt user (uid 42) while fetching.
# Under rootless Podman with a single UID mapping that setuid fails and apt dies with
# "Method http has died unexpectedly". Set as config rather than a flag because both
# `update` and `install` need it. Removed again afterwards; a no-op on a Docker host.
RUN printf 'APT::Sandbox::User "root";\n' > /etc/apt/apt.conf.d/00no-sandbox \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libxcb1 \
    && rm -f /etc/apt/apt.conf.d/00no-sandbox \
    && rm -rf /var/lib/apt/lists/*

# /opt/app is created here, as root, and never chowned. Rootless Podman on a host with
# no /etc/subuid range can only map one UID, so any chown to a second UID fails the
# build; nothing in this image needs to be owned by the unprivileged user, only read
# by it. See SUBMISSION_NOTES.md, "Building here".
RUN mkdir -p /opt/app
WORKDIR /opt/app

COPY requirements.txt /opt/app/

RUN python -m pip install \
    --break-system-packages \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

# Fail the BUILD rather than the submission. A torch built for the wrong CUDA does not
# raise -- torch.cuda.is_available() just returns False and the model runs on CPU.
# Both halves matter: the CUDA version is what the driver floor depends on and is NOT
# visible in the architecture list, while the arch list is what an older 12.4/12.6
# wheel gets wrong.
RUN python -c "import torch; \
flags = torch._C._cuda_getArchFlags() or ''; \
cuda = torch.version.cuda or ''; \
print('torch', torch.__version__, '| cuda', cuda, '|', flags); \
assert cuda.startswith('12.8'), 'torch was replaced by a CUDA ' + cuda + ' build; the L40S driver cannot run it'; \
assert 'sm_120' in flags and 'sm_86' in flags, 'torch build misses a required GPU architecture: ' + repr(flags)"

# Everything the container imports, proven before any weights are copied.
RUN python -c "import transformers, peft, av, numpy, PIL, focus; \
from transformers import Qwen3_5ForConditionalGeneration; \
from focus import Request, Response, load_requests, save_items; \
from focus.foreign_objects import FOType; \
print('transformers', transformers.__version__, '| peft', peft.__version__, '| av', av.__version__); \
names = FOType.names(); \
assert names == ('Sponge', 'Clip', 'Specimen Bag', 'Silicone Loop', 'External Drain', 'Needle', 'Gallstone', 'Specimen', 'Mesh', 'Absorbable Hemostatic Agent'), names; \
print('FO classes match training:', len(names))"



FROM deps AS procedure-algorithm-amd64

# Base weights, ~52 GB across fifteen shards, SPLIT ACROSS THREE LAYERS.
#
# The split is required, not cosmetic: the challenge states "a single image layer must
# not exceed 50 GB", and one COPY of this directory produces a 52 GB layer. Five shards
# per layer is ~20 GB each, which leaves margin whether the platform measures the limit
# on the compressed blob or the extracted tree.
#
# These layers are byte-identical across the three epoch images, so Podman stores them
# once and only the adapter below differs.
COPY resources/base_model/model-0000[1-5]-of-00015.safetensors /opt/app/resources/base_model/
COPY resources/base_model/model-0000[6-9]-of-00015.safetensors \
     resources/base_model/model-00010-of-00015.safetensors      /opt/app/resources/base_model/
COPY resources/base_model/model-0001[1-5]-of-00015.safetensors /opt/app/resources/base_model/

# Config, tokenizer and the shard index -- a few MB, and below the weights so that
# editing one does not invalidate 52 GB of layers above it.
COPY resources/base_model/*.json  /opt/app/resources/base_model/
COPY resources/base_model/*.jinja /opt/app/resources/base_model/
COPY resources/base_model/*.txt   /opt/app/resources/base_model/

# The code. It is small and changes often, so it sits BELOW the weights: a layer change
# invalidates everything after it, so code above the weights would force all 52 GB to be
# re-copied on every code edit.
COPY resources/*.py /opt/app/resources/

# The LoRA adapter for this epoch (~58 MB), merged into the base at load.
COPY resources/adapter/ /opt/app/resources/adapter/

COPY inference.py /opt/app/

# This base image runs Python 3.11; the training tree runs 3.13. Byte-compiling every
# shipped module turns a syntax feature the older interpreter lacks into a build error
# rather than a crash on question 1.
RUN python -m compileall -q /opt/app/inference.py /opt/app/resources

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

ENTRYPOINT ["python", "inference.py"]
