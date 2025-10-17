# Variables
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
DEVICE="cuda"
DATASETS="ag_news" #Could be changed to 'toxic_text', 'twitter_emotion'
BS=16
SHOT_MIN=8
SHOT_MAJ=8
MAJORITY_LABEL="sports" #This label is in agnews
MAX_TOKENS=3

# Self-Consistency Parameters
USE_SC=true  
SC_SAMPLES=5 
SC_TEMP=0.7 

# Run Python script
if [ "$USE_SC" = true ]; then
    echo "Running with SELF-CONSISTENCY mode (samples=$SC_SAMPLES, temp=$SC_TEMP)"
    python3 src/eval_llm.py --model $MODEL \
                                --device $DEVICE \
                                --datasets $DATASETS \
                                --different-shots \
                                --batch-size $BS \
                                --shots-minority $SHOT_MIN \
                                --shots-majority $SHOT_MAJ \
                                --majority-label $MAJORITY_LABEL \
                                --max-tokens $MAX_TOKENS \
                                --use-self-consistency \
                                --sc-samples $SC_SAMPLES \
                                --sc-temperature $SC_TEMP
else
    echo "Running with GREEDY DECODING mode"
    python3 src/eval_llm.py --model $MODEL \
                                --device $DEVICE \
                                --datasets $DATASETS \
                                --different-shots \
                                --batch-size $BS \
                                --shots-minority $SHOT_MIN \
                                --shots-majority $SHOT_MAJ \
                                --majority-label $MAJORITY_LABEL \
                                --max-tokens $MAX_TOKENS
fi