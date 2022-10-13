from utils import convert_to_json
from metric.evaluator import get_evaluator

# Example for data-to-text
task = 'data2text'

# a list of model outputs to be evaluataed
output_list = ['You would like to search financial district ?']
# a list of human-annotated reference texts
ref_list = ['You are looking near the financial district , right ?']

# Prepare data for pre-trained evaluators
data = convert_to_json(output_list=output_list, ref_list=ref_list)
# Initialize evaluator for a specific task
evaluator = get_evaluator(task)
# Get multi-dimensional evaluation scores
eval_scores = evaluator.evaluate(data, print_result=True)



'''
# Example for summarization
task = 'summarization'

# a list of source documents
src_list = ['Peter and Elizabeth took a taxi to attend the night party in the city. \
             While in the party, Elizabeth collapsed and was rushed to the hospital.']
# a list of human-annotated reference summaries
ref_list = ['Elizabeth was hospitalized after attending a party with Peter.']
# a list of model outputs to be evaluataed
output_list = ['Peter and Elizabeth attend party city. Elizabeth rushed hospital.']

# Prepare data for pre-trained evaluators
data = convert_to_json(output_list=output_list, 
                       src_list=src_list, ref_list=ref_list)
# Initialize evaluator for a specific task
evaluator = get_evaluator(task)
# Get multi-dimensional evaluation scores
eval_scores = evaluator.evaluate(data, print_result=True)
# eval_scores = evaluator.evaluate(data, dims=['coherence', 'consistency', 'fluency'], 
#                                  overall=False, print_result=True)




# Example for dialogue response generation
task = 'dialogue'

# a list of dialogue histories
src_list = ['hi , do you know much about the internet ? \n i know a lot about different sites and some website design , how about you ? \n\n']
# a list of additional context that should be included into the generated response
context_list = ['the 3 horizontal line menu on apps and websites is called a hamburger button .\n']
# a list of model outputs to be evaluated
output_list = ['i do too . did you know the 3 horizontal line menu on apps and websites is called the hamburger button ?']

# Prepare data for pre-trained evaluators
data = convert_to_json(output_list=output_list, 
                       src_list=src_list, context_list=context_list)
# Initialize evaluator for a specific task
evaluator = get_evaluator(task)
# Get multi-dimensional evaluation scores
eval_scores = evaluator.evaluate(data, print_result=True)



# Example for factual consistency detection
task = 'fact'

# a list of source documents
src_list = ['Peter and Elizabeth took a taxi to attend the night party in the city. \
             While in the party, Elizabeth collapsed and was rushed to the hospital.']
# a list of model outputs (claims) to be evaluataed
output_list = ['Tom was rushed to hospital.']

# Prepare data for pre-trained evaluators
data = convert_to_json(output_list=output_list, src_list=src_list)
# Initialize evaluator for a specific task
evaluator = get_evaluator(task)
# Get factual consistency scores
eval_scores = evaluator.evaluate(data, print_result=True)
'''