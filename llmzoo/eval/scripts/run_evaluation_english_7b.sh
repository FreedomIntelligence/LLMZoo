#!/bin/bash


eva=order # score / order

PROJECT_PATH=/to_path/LLMZoo/llmzoo/eval
PROMPT_PATH=$PROJECT_PATH/prompts
QUESTION_PATH=$PROJECT_PATH/questions
ANSWER_PATH=$PROJECT_PATH/output
REVIEW_PATH=$PROJECT_PATH/review_en_7b

models=(Chimera-chat-7b Phoenix-chat-7b Vicuna-7b)



# ------------------------- EVALUATION -------------------------
lang_list=(en)
general_dimensions=(general relevance coherence diversity)
role_dimension=immersion
dimension_list=(${general_dimensions[*]} $role_dimension)

rule=$PROMPT_PATH/$eva/prompt_all.json


# load answer files
i=1
answer_list=($answer_gpt)
for model in ${models[*]}; do
        answer_file=$ANSWER_PATH/answer_$model.jsonl;
        answer_list[i]=$answer_file
        i=`expr $i + 1`
done


mkdir $REVIEW_PATH
touch $REVIEW_PATH/model.txt
result=""
for element in "${models[@]}"; do result+="$element\n"; done
echo -e "$result" > $REVIEW_PATH/model.txt



echo "Start Evaluation"
for lang in ${lang_list[*]}; do
        cur_lang_path=$REVIEW_PATH/$lang
        mkdir $cur_lang_path

        general_question=$QUESTION_PATH/questions-$lang.jsonl
        role_question=$QUESTION_PATH/roleplay-questions-$lang.jsonl

        # load question files
        question_list=()
        for i in $(seq 0 1 `expr ${#general_dimensions[*]} - 1`); do
                question_list[i]=$general_question
        done
        question_list=(${question_list[*]} $role_question)

        for i in $(seq 0 1 `expr ${#dimension_list[*]} - 1`); do
                question=${question_list[i]}
                dimension=${dimension_list[i]}
                cur_review_path=$cur_lang_path/$dimension
                mkdir $cur_review_path

                if [[ $eva = "score" ]]; then
                        python $PROJECT_PATH/eval_gpt_review_all.py \
                                --question $question \
                                --answer-list ${answer_list[*]}  \
                                --rule $rule \
                                --dimension $dimension \
                                --output $cur_review_path/review.jsonl
                else
                        python $PROJECT_PATH/eval_gpt_review_all.py \
                                --question $question \
                                --answer-list ${answer_list[*]}  \
                                --rule $rule \
                                --dimension $dimension \
                                --output $cur_review_path/review.jsonl \
                                --order
                fi

                if [[ $? = "127" ]]; then exit; fi
                echo Output to $cur_review_path/review.jsonl
        done

done
echo "Finish"




















