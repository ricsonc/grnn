#!/usr/bin/env bash

ssh -N -4 -L :9001:localhost:9001 compute-0-$1
