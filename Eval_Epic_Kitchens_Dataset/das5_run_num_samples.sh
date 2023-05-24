# Runs ablation on number of training samples

# model="R2PLUS1D"
# base_cfg_name="32x112x112_R18_K400_LR0.0025.yaml"
# model="GDT"
# base_cfg_name="32x112x112_R2+1D_K400_LR0.0025.yaml"
# model="PRETEXT_CONTRAST"
# base_cfg_name="32x112x112_PC_R18_K400_LR0.0025.yaml"
# model="SELAVI"
# base_cfg_name="32x112x112_R18_K400_LR0.0025.yaml"
model="TCLR"
base_cfg_name="32x112x112_R18_K400_LR0.0025_no_norm.yaml"
for num in {1000,2000,4000,8000,16000,32000}
do
    echo "-------------------------- Running $model with $num samples --------------------"
    # cfg="configs/EPIC-KITCHENS/$model/n_samples_"$num"_"$base_cfg_name
    cfg="configs/EPIC-KITCHENS/$model/das5_n_samples_"$num"_"$base_cfg_name

    echo "Config: "
    ls $cfg
    echo ""

    bash das5_train.sh -c $cfg
    bash das5_val.sh -c $cfg

    echo "-------------------------- xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx --------------------"
    echo ""
done
