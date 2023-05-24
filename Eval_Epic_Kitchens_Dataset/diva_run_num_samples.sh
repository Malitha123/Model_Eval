# Runs ablation on number of training samples

# model="VIDEOMOCO"
# model="AVID_CMA"
# model="CTP"
# base_cfg_name="32x112x112_R2+1D-18_K400_LR0.0025.yaml"
# model="RSPNET"
# base_cfg_name="32x112x112_R18_K400_LR0.0025.yaml"
# model="MOCO"
# base_cfg_name="32x112x112_R18_K400_LR0.00025.yaml"
model="TCLR"
base_cfg_name="32x112x112_R18_K400_LR0.0025_no_norm.yaml"
for num in {1000,2000,4000,8000,16000,32000}
do
    echo "-------------------------- Running $model with $num samples --------------------"
    cfg="configs/EPIC-KITCHENS/$model/diva_n_samples_"$num"_"$base_cfg_name

    echo "Config: "
    ls $cfg
    echo ""

    bash diva_train.sh -c $cfg
    bash diva_val.sh -c $cfg

    echo "-------------------------- xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx --------------------"
    echo ""
done
