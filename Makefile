train_ref:
	python train.py --env-name pretrain

train_R2:
	python train.py --env-name pretrain-R2 --sim-reward R2

train_nofilm:
	python train.py --env-name pretrain-nofilm --sim-no-static


experiment_g2_compare:
	python train.py --env-name comp2_ref --gset 2 --num-env-steps 51200000 --snapshot pretrain-30000; \
	python train.py --env-name comp2_R2 --gset 2 --num-env-steps 51200000 --snapshot pretrain-R2-30000; \
	python train.py --env-name comp2_scratch --gset 2 --num-env-steps 51200000; \
	python train.py --env-name comp2_R2_scratch --gset 2 --num-env-steps 51200000 --sim-reward R2;


experiment_ref:
	for G in 1 2 3 4 5 6 7 8 9 10; do \
		python train.py --env-name ref_G$${G} --gset $${G} --num-env-steps 12800000 --snapshot pretrain-30000; \
	done

experiment_R2:
	for G in 1 2 3 4 5 6 7 8 9 10; do \
		python train.py --env-name R2_G$${G} --gset $${G} --num-env-steps 12800000 --snapshot pretrain-R2-30000; \
	done

experiment_scratch:
	for G in 1 2 3 4 5 6 7 8 9 10; do \
		python train.py --env-name scratch_G$${G} --gset $${G} --num-env-steps 12800000; \
	done

experiment_R2_scratch:
	for G in 1 2 3 4 5 6 7 8 9 10; do \
		python train.py --env-name R2_scratch_G$${G} --gset $${G} --num-env-steps 12800000 --sim-reward R2; \
	done

experiment_nofilm:
	for G in 1 2 3 4 5 6 7 8 9 10; do \
		python train.py --env-name nofilm_G$${G} --gset $${G} --num-env-steps 12800000 --snapshot pretrain-nofilm-30000; \
	done

experiment_nofilm_scratch:
	for G in 1 2 3 4 5 6 7 8 9 10; do \
		python train.py --env-name nofilm_G$${G} --gset $${G} --num-env-steps 12800000 --sim-no-static; \
	done




