ACTION=$1
case $ACTION in
    'walk')
        python -m python_scripts.evaluation.cal_metric_locomotion
        ;;
    'sit')
        python -m python_scripts.evaluation.cal_metric_interaction --action "sit"
        ;;
    'lie')
        python -m python_scripts.evaluation.cal_metric_interaction --action "lie"
        ;;
    *)
        echo "unknown action: $ACTION"
        exit 1
        ;;
esac
