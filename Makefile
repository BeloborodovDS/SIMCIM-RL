train_ref:
	for s in 1 2 3; do \
		python train.py --env-name pretrain-seed$${s} --seed $${s} --num-env-steps 800000000; \
	done

train_R2:
	for s in 1 2 3; do \
		python train.py --env-name pretrain-R2-seed$${s} --seed $${s} --sim-reward R2 --num-env-steps 800000000; \
	done

train_nofilm:
	for s in 1 2 3; do \
		python train.py --env-name pretrain-nofilm-seed$${s} --seed $${s} --sim-no-static --num-env-steps 800000000; \
	done


experiment_ref:
	for s in 1 2 3; do \
	for G in 1 2 3 4 5 6 7 8 9 10; do \
		python train.py --env-name ref-G$${G}-seed$${s} --gset $${G} --seed $${s} --num-env-steps 25600000 --snapshot pretrain-seed$${s}-30000; \
	done; \
	done

experiment_R2:
	for s in 1 2 3; do \
	for G in 1 2 3 4 5 6 7 8 9 10; do \
		python train.py --env-name R2-G$${G}-seed$${s} --gset $${G} --seed $${s} --num-env-steps 25600000 --snapshot pretrain-R2-seed$${s}-30000; \
	done; \
	done

experiment_nofilm:
	for s in 1 2 3; do \
	for G in 1 2 3 4 5 6 7 8 9 10; do \
		python train.py --env-name nofilm-G$${G}-seed$${s} --gset $${G} --seed $${s} --num-env-steps 25600000 --snapshot pretrain-nofilm-seed$${s}-30000; \
	done; \
	done


experiment_scratch:
	for s in 1 2 3; do \
	for G in 1 2 3 4 5 6 7 8 9 10; do \
		python train.py --env-name scratch-G$${G}-seed$${s} --gset $${G} --seed $${s} --num-env-steps 25600000; \
	done; \
	done

experiment_R2_scratch:
	for s in 1 2 3; do \
	for G in 1 2 3 4 5 6 7 8 9 10; do \
		python train.py --env-name R2-scratch-G$${G}-seed$${s} --gset $${G} --seed $${s} --num-env-steps 25600000 --sim-reward R2; \
	done; \
	done

experiment_nofilm_scratch:
	for s in 1 2 3; do \
	for G in 1 2 3 4 5 6 7 8 9 10; do \
		python train.py --env-name nofilm-scratch-G$${G}-seed$${s} --gset $${G} --seed $${s} --num-env-steps 25600000 --sim-no-static; \
	done; \
	done




