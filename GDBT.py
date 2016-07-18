__author__ = 'raoqiang'
import graphlab as gl
# load training data
training_sframe = gl.SFrame.read_csv('train.csv')

# train a model
features = ['datetime', 'season', 'holiday', 'workingday', 'weather',
            'temp', 'atemp', 'humidity', 'windspeed']
#m = gl.boosted_trees_regression.create(training_sframe,'count',
#                            feature_columns=features,
#                            target_column='count', objective='regression',
#                            num_iterations=100)

m = gl.boosted_trees_regression.create(training_sframe,'count',features=features,max_iterations=10)



# predict on test data
test_sframe = gl.SFrame.read_csv('test.csv')
prediction = m.predict(test_sframe)
def make_submission(prediction, filename='result1.txt'):
    with open(filename, 'w') as f:
        f.write('datetime,count')
        submission_strings = test_sframe['datetime'] + ',' + prediction.astype(str)
        for row in submission_strings:
            f.write(row + '\n')

make_submission(prediction, 'submission1.txt')


from datetime import datetime
date_format_str = '%Y-%m-%d %H:%M:%S'

def parse_date(date_str):
    """Return parsed datetime tuple"""
    d = datetime.strptime(date_str, date_format_str)
    return {'year': d.year, 'month': d.month, 'day': d.day,
            'hour': d.hour, 'weekday': d.weekday()}

def process_date_column(data_sframe):
    """Split the 'datetime' column of a given sframe"""
    parsed_date = data_sframe['datetime'].apply(parse_date).unpack(column_name_prefix='')
    for col in ['year', 'month', 'day', 'hour', 'weekday']:
        data_sframe[col] = parsed_date[col]

process_date_column(training_sframe)
process_date_column(test_sframe)

import math

# Create three new columns: log-casual, log-registered, and log-count
for col in ['casual', 'registered', 'count']:
    training_sframe['log-' + col] = training_sframe[col].apply(lambda x: math.log(1 + x))


    new_features = features + ['year', 'month', 'weekday', 'hour']
new_features.remove('datetime')

m1 = gl.boosted_trees_regression.create(training_sframe,target='log-casual',
                             features=new_features,
                             max_iterations=50)

m2 = gl.boosted_trees_regression.create(training_sframe,target='log-registered',
                             features=new_features,
                             max_iterations=50)

def fused_predict(m1, m2, test_sframe):
    """
    Fused the prediction of two separately trained models.
    The input models are trained in the log domain.
    Return the combine predictions in the original domain.
    """
    p1 = m1.predict(test_sframe).apply(lambda x: math.exp(x)-1)
    p2 = m2.predict(test_sframe).apply(lambda x: math.exp(x)-1)
    return (p1 + p2).apply(lambda x: x if x > 0 else 0)

prediction = fused_predict(m1, m2, test_sframe)


env = gl.deploy.environment.Local('hyperparam_search')


training = training_sframe[training_sframe['day'] <= 16]
validation = training_sframe[training_sframe['day'] > 16]

try:
    training.save('/tmp/training')
    validation.save('/tmp/validation')
    print 'save successfully'
except Exception, e:
    print e


#for a in training:
#print str(a)
print 'hello'

ntrees = 500
search_space = {
    'max_depth': [4,5,6,7,8,9,10,12],
    'min_child_weight': [16,18, 20,22,24,26,28],

    'step_size': [0.02,0.04,0.05],
    'max_iterations': ntrees
}
'''

search_space = {
    '**kwargs': {
        'max_depth': [10, 15, 20],
        'min_child_weight': [5, 10, 20]
    },
    'step_size': 0.05,
    'max_iterations': ntrees
}

'''
def parameter_search(training_url, validation_url, default_params):
    """
    Return the optimal parameters in the given search space.
    The parameter returned has the lowest validation rmse.
    """

    job = gl.toolkits.model_parameter_search(gl.boosted_trees_regression.create,
                                             train_set_path=training_url,
                                             save_path='/tmp/job_output',
                                             test_set_path=validation_url,
                                             standard_model_params=default_params,
                                             hyper_params=search_space,
                                             max_num_models='all',
                                             environment =env)


    # When the job is done, the result is stored in an SFrame
    # The result contains attributes of the models in the search space
    # and the validation error in RMSE.
    print gl.SFrame('/tmp/job_output')
    print gl.SFrame('/tmp/job_output')['parameters']
    print gl.SFrame('/tmp/job_output')['test_metrics'][3]

    '''
    result = gl.SFrame('../tmp/job_output').sort('rmse', ascending=True)
    gl.SFrame.sort(gl.SFrame('../tmp/job_output')['test_metrics'])
    #Return the parameters with the lowest validation error.
    optimal_params = result[['max_depth', 'min_child_weight']][0]
    optimal_rmse = result['rmse'][0]
    print('Optimal parameters: %s' % str(optimal_params))
    print('RMSE: %s' % str(optimal_rmse))
    return optimal_params
    '''
    result = gl.SFrame('/tmp/job_output')
    index=0
    b=result['test_metrics'][0]['rmse']
    for j in range(0,len(result['test_metrics'])):
        if(b>result['test_metrics'][j]['rmse']):
            b=result['test_metrics'][j]['rmse']
            index=j

    optimal_rmse = result['test_metrics'][index]['rmse']
    optimal_params=[result['parameters'][index]['max_depth'],result['parameters'][index]['min_child_weight']]
    print('Optimal parameters: %s' % str(optimal_params))
    print('RMSE: %s' % str(optimal_rmse))
    return optimal_params

fixed_params = {'features': new_features,
                'verbose': False}

fixed_params['target'] = 'log-casual'
params_log_casual = parameter_search(training_url='/tmp/training',
                                     validation_url='/tmp/validation',
                                     default_params=fixed_params)

fixed_params['target'] = 'log-registered'
params_log_registered = parameter_search('/tmp/training',
                                         '/tmp/validation',
                                         fixed_params)






'''
create(dataset, target,
           features=None, max_iterations=10,
           validation_set=None,
           verbose=True,
           **kwargs):
'''


m_log_registered = gl.boosted_trees_regression.create(training_sframe,
                                           features=new_features,
                                           target='log-registered',
                                           max_iterations=ntrees,
                                           verbose=False,
                                           max_depth=params_log_registered[0],
                                           min_child_weight=params_log_registered[1])

m_log_casual = gl.boosted_trees_regression.create(training_sframe,
                                       features=new_features,
                                       target='log-casual',
                                       max_iterations=ntrees,
                                       verbose=False,
                                       max_depth=params_log_casual[0],
                                       min_child_weight=params_log_casual[1])

final_prediction = fused_predict(m_log_registered, m_log_casual, test_sframe)

make_submission(final_prediction, 'result2.txt')



