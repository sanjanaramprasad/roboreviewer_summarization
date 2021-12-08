import pandas as pd 


df = pd.read_csv('/home/ramprasad.sa/roboreviewer_summarization/scripts/bart_multi_lm/run_inference_output_lm_ind_weights.csv')

model_outputs = df['Generated Summary']
refs = df['Reference Summary']
pop = df['Population']
ints = df['Interventions']
outs = df['Outcomes']
ptext = df['Punchline Texts']
with open('trial_output.txt' , 'w') as fp:
	all_lines = "\n".join(model_outputs)
	fp.write(all_lines)


with open('trial_ref.txt' , 'w') as fp:
        all_lines = "\n".join(refs)
        fp.write(all_lines)


def write(filename, lines):
     with open(filename, 'w') as fp:
        all_lines = "\n".join(lines)
        fp.write(all_lines)


write('trial_pop.txt', pop)
write('trial_int.txt', ints)
write('trial_ptext.txt', ptext)
write('trial_out.txt', outs)
