#!/bin/bash

horiz=True

eva=order # score / order

PROJECT_PATH=/to_path/LLMZoo/llmzoo/eval
QUESTION_PATH=$PROJECT_PATH/questions
REVIEW_PATH=$PROJECT_PATH/review_output



# ------------
lang_list=(en zh)
general_dimensions=(general relevance coherence diversity)
role_dimension=immersion
dimension_list=(${general_dimensions[*]} $role_dimension)



echo "Start"
for lang in ${lang_list[*]}; do
        cur_lang_path=$REVIEW_PATH/$lang

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

                if [[ $eva = "score" ]]; then
                        if [[ $horiz = "True" ]]; then
                                python $PROJECT_PATH/compute_metric_all.py \
                                        --question $question \
                                        --review $cur_review_path/review.jsonl \
                                        --output $cur_review_path/metric.json \
                                        --horiz
                        else
                                python $PROJECT_PATH/compute_metric_all.py \
                                        --question $question \
                                        --review $cur_review_path/review.jsonl \
                                        --output $cur_review_path/metric.json \            
                        fi
                        
                else
                        if [[ $horiz = "True" ]]; then
                                python $PROJECT_PATH/compute_metric_all.py \
                                        --question $question \
                                        --review $cur_review_path/review.jsonl  \
                                        --output $cur_review_path/metric.json \
                                        --order \
                                        --horiz

                                python $PROJECT_PATH/summary_ordering.py \
                                        --review $cur_review_path/metric.json \
                                        --output $cur_review_path/ordering.txt \
                                        --horiz

                        else
                                python $PROJECT_PATH/compute_metric_all.py \
                                        --question $question \
                                        --review $cur_review_path/review.jsonl  \
                                        --output $cur_review_path/metric.json \
                                        --order

                                python $PROJECT_PATH/summary_ordering.py \
                                        --review $cur_review_path/metric.json \
                                        --output $cur_review_path/ordering.txt
                        fi
                        
                fi

                echo "Done $cur_review_path"
        done

done

echo "Finish"





