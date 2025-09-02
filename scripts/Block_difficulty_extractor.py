# Extracting the n-back type for a single block...
def extract_nblock(sequence, targets, zero_flag):
    for name, arg in {"sequence": sequence, "targets": targets}.items():
        if not isinstance(arg, list):
            raise TypeError(f"'{name}' is not a list (got: {type(arg).__name__})")
    
    # Special Handling of Block 0...    
    if zero_flag:
        return 0
    
    n_vals = np.zeros(4)
    for t in targets:
        target_letter = sequence[t]
        if target_letter == sequence[t-1]:
           n_vals[1] += 1
        if target_letter == sequence[t-2]:
           n_vals[2] += 1
        if target_letter == sequence[t-3]:
           n_vals[3] += 1
    return np.argmax(n_vals)

# Extracting the n-back type for each block...
def calculate_nvals(df):
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas.DataFrame, but got: {type(df).__name__}")
    
    # Sequences
    mask_seq = df['marker'].str.startswith("sequence")
    seq_df = (
        df.loc[mask_seq, "marker"]
        .str.removeprefix("sequence_")     
        .str.split(",")                    
        .to_frame(name="sequence")         
        .reset_index(drop=True)            
    )
    #print(seq_df)

    # Targets
    mask_trg = df['marker'].str.startswith("targets")
    trg_df = (
        df.loc[mask_trg, "marker"]
        .str.removeprefix("targets_")     
        .str.split(",")    
        .apply(lambda x: [int(i) for i in x])   # Change to Integer...                
        .to_frame(name="targets")         
        .reset_index(drop=True)            
    )
    #print(trg_df)

    n_vals = []
    for idx in range(len(seq_df)):
        seq = seq_df.at[idx, 'sequence']
        trg = trg_df.at[idx, 'targets']
        if idx == 0:
            n_vals.append(int(extract_nblock(seq, trg, True)))
        else:
            n_vals.append(int(extract_nblock(seq, trg, False)))

    return n_vals