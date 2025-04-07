#!/bin/bash
echo "Running Linear"
python3 ppo.py --policy_path "./model/03.20-21:14:53-linear-policy.pt" --output_name "linear"
echo "Running Cosine"
python3 ppo.py --policy_path "./model/03.21-05:57:33-cosine_with_restarts-policy.pt" --output_name "cosine_with_restarts"
echo "Running Polynomial"
python3 ppo.py --policy_path "./model/03.21-10:18:28-polynomial-policy.pt" --output_name "polynomial"
python3 benchmark.py