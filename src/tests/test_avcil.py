from tests import run_main_and_assert

FAST_LOCAL_TEST_ARGS = "--exp-name local_test_avcil --datasets VGGSound_balance" \
                       " --network avcil_net --num-tasks 3 --seed 1 --batch-size 32" \
                       " --nepochs 3 --num-workers 0 --approach avcil"


def test_avcil_without_exemplars():
    run_main_and_assert(FAST_LOCAL_TEST_ARGS)


def test_avcil_with_exemplars():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --num-exemplars 200"
    args_line += " --instance-contrastive --class-contrastive --attn-score-distil"
    run_main_and_assert(args_line)


def test_avcil_with_warmup():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --warmup-nepochs 5"
    args_line += " --warmup-lr-factor 0.5"
    args_line += " --num-exemplars 200"
    run_main_and_assert(args_line)


def test_avcil_with_instance_contrastive():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --num-exemplars 200"
    args_line += " --instance-contrastive"
    run_main_and_assert(args_line)


def test_avcil_with_class_contrastive():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --num-exemplars 200"
    args_line += " --class-contrastive"
    run_main_and_assert(args_line)


def test_avcil_with_attn_score_distil():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --num-exemplars 200"
    args_line += " --attn-score-distil"
    run_main_and_assert(args_line)