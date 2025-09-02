#!/usr/bin/env python3
import csv
import random
from psychopy import visual, core, event, gui
from pylsl import StreamInfo, StreamOutlet, local_clock

# ────────────────────────────────────────────────────────────────────────
#                              EXPERIMENT PARAMETERS                       
# ────────────────────────────────────────────────────────────────────────

SYMBOLS            = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# trials = no. of smybols shown per block
BASELINE_TRIALS    = 48
PRACTICE_TRIALS    = 20
MAIN_TRIALS        = 48
MAIN_BLOCKS        = 6

BREAK_DURATION     = 30    # seconds, adjustable break between main blocks

FIXATION_DUR       = 2.0
STIM_DUR           = 1.0
FEEDBACK_DUR       = .5

COLOR_FIXATION     = "white"
COLOR_CORRECT      = "green"
COLOR_INCORRECT    = "red"

LSL_NAME           = "NBackMarkers"
LSL_TYPE           = "Markers"
LSL_ID             = "nback_stream_001"

# ────────────────────────────────────────────────────────────────────────
#                             GLOBAL MARKER LOGGING                             
# ────────────────────────────────────────────────────────────────────────

# will hold tuples of (marker_string, timestamp)
marker_log = []

def log_marker(marker_str):
    """
    Pushes an LSL marker with its timestamp, and logs both to marker_log.csv
    file as redundancy/fail-safe for post-hoc corrections.
    """
    ts = local_clock()
    outlet.push_sample([marker_str], ts)
    marker_log.append((marker_str, ts))

# ────────────────────────────────────────────────────────────────────────
#                          UTILITY FUNCTIONS                             
# ────────────────────────────────────────────────────────────────────────

def check_for_quit(win):
    if event.getKeys(keyList=["escape"]):
        win.close()
        core.quit()

def show_break(win, block_num, total_blocks, duration=BREAK_DURATION):
    """
    Shows a break screen that tells the participant:
      “Block {block_num} of {total_blocks} complete.
       Resuming in {secs} sec”
    """
    timer = core.CountdownTimer(duration)
    text  = visual.TextStim(
        win,
        text="",            # we’ll set text inside the loop
        color="white",
        wrapWidth=1.2,
        height=0.06,
        pos=(0, 0)
    )

    while timer.getTime() > 0:
        secs = int(timer.getTime())
        text.text = (
            f"Block {block_num} of {total_blocks} complete\n\n"
            f"Resuming in {secs} sec"
        )
        text.draw()
        win.flip()
        core.wait(1.0)
        check_for_quit(win)

    # after countdown ends, show questions
    ask_questions(win)

def generate_n_back_sequence(n, length, symbols=SYMBOLS):
    """
    Builds an n-back sequence of given length with exactly 50% targets.
    Returns (seq, targets, target_symbol) where target_symbol is only
    non-None for n=0.
    """
    if n == 0:
        target_symbol   = random.choice(symbols)
        # choosing 33 % to be targets, can be changed
        num_targets     = length // 3
        target_positions = set(random.sample(range(length), num_targets))
        seq, targets = [], []
        for i in range(length):
            if i in target_positions:
                seq.append(target_symbol)
                targets.append(i)
            else:
                non_t = [s for s in symbols if s != target_symbol]
                seq.append(random.choice(non_t))
        return seq, targets, target_symbol

    seq, targets = [], []
    all_pos       = list(range(n, length))
    num_targets   = (length - n) // 3
    targ_pos      = set(random.sample(all_pos, num_targets))

    for i in range(length):
        if i < n:
            seq.append(random.choice(symbols))
        else:
            if i in targ_pos:
                seq.append(seq[i - n])
                targets.append(i)
            else:
                avoid = seq[i - n]
                pool  = [s for s in symbols if s != avoid]
                seq.append(random.choice(pool))

    return seq, targets, None


def ask_questions(win):
    """
    Presents a set of interactive questions after a block.
    Returns a dictionary of responses.
    """
    responses = {}

    questions = [
        ("How focused did you feel?\n(1 = not at all, 5 = very focused)", ["1","2","3","4","5"]),
        ("How difficult was this block?\n(1 = very easy, 5 = very hard)", ["1","2","3","4","5"]),
        ("Do you feel ready to continue?\n(y = yes, n = no)", ["y","n"])
    ]

    for qtext, valid_keys in questions:
        stim = visual.TextStim(
            win, text=qtext, color="white", wrapWidth=1.2, height=0.06
        )
        stim.draw(); win.flip()
        key = event.waitKeys(keyList=valid_keys + ["escape"])[0]
        if key == "escape":
            win.close(); core.quit()
        responses[qtext] = key

        # log via LSL and marker_log redundancy
        log_marker(f"question_{qtext.replace(' ','_')}_resp_{key}")

    return responses


# ────────────────────────────────────────────────────────────────────────
#                         RUN ONE PRACTICE/MAIN BLOCK                      
# ────────────────────────────────────────────────────────────────────────

def run_block(win, outlet_handle, n, length, practice=False, block_num=1):
    """
    Runs a single n-back block, sending & logging markers for:
      - block start/end
      - sequence & target list
      - each trial (onset, resp, correctness, RT, end)
      - block accuracy
    Returns block accuracy (float) for main blocks, else None.
    """
    block_type = "practice" if practice else "main"

    # 1) Instructions
    header = "0-back BASELINE" if n == 0 else f"{'Practice ' if practice else ''}{n}-back"
    instr = visual.TextStim(
        win,
        text=f"{header}\n\nWhen you see the target, press SPACEBAR, otherwise don't press anything.\n\nPress any key to start.",
        color="white",
        wrapWidth=1.2,
        height=0.06
    )
    instr.draw(); win.flip()
    event.waitKeys(); check_for_quit(win)

    # 2) Generate sequence
    seq, targets, target_symbol = generate_n_back_sequence(n, length)

    # 2a) Preview 0-back target
    if n == 0:
        lbl = visual.TextStim(win, text="Target Symbol",
                              color="white", pos=(0, 0.2), height=0.06)
        sym = visual.TextStim(win, text=target_symbol,
                              color="white", pos=(0, -0.1), height=0.2)
        lbl.draw(); sym.draw()
        win.flip(); core.wait(3.0)
        check_for_quit(win)

    # 3) Block start markers
    log_marker(f"{block_type}_block_{block_num}_start")
    log_marker(f"sequence_{','.join(seq)}")
    log_marker(f"targets_{','.join(str(i) for i in targets)}")

    # Prepare stimuli & counters
    fixation = visual.TextStim(win, text="+", height=0.06, color=COLOR_FIXATION)
    stim     = visual.TextStim(win, text="", height=0.18, color="white")
    feedback = visual.Rect(win, width=1.0, height=1.0, lineColor=None)
    margin   = 0.02
    correct_count = 0

    # 4) Trials
    for i, char in enumerate(seq):
        check_for_quit(win)
        log_marker(f"{block_type}_block_{block_num}_trial_{i}_on")

        fixation.draw(); win.flip()
        core.wait(FIXATION_DUR)

        stim.text = char
        stim.draw(); win.flip()
        
        # UPDATE: Converting to go-no-go task instead
        resp, rt = None, None
        clk = core.Clock()
        while clk.getTime() < STIM_DUR:
            keys = event.getKeys(keyList=["space", "escape"])
            if "escape" in keys:
                win.close(); core.quit()
            if "space" in keys:
                resp, rt = True, clk.getTime()
                break
        # If no response, resp stays None (miss)
        if resp is None:
            resp = False

        is_targ = (i in targets)
        correct = None if resp is None else (resp == is_targ)
        if correct:
            correct_count += 1

        log_marker(f"resp_{resp}")
        log_marker(f"corr_{correct}")
        if rt is not None:
            log_marker(f"rt_{rt:.3f}")

        # UPDATE:  Show feedback only in practice blocks
        if practice:
            stim.draw()
            bb_w, bb_h = stim.boundingBox
            feedback.width  = bb_w + margin
            feedback.height = bb_h + margin
            feedback.pos    = stim.pos
            feedback.fillColor = (
                COLOR_CORRECT   if correct
                else COLOR_INCORRECT if correct is False
                else None
            )
            feedback.draw(); stim.draw()
            win.flip(); core.wait(FEEDBACK_DUR)

        log_marker(f"trial_{i}_end")

    # 5) Block end & accuracy
    log_marker(f"{block_type}_block_{block_num}_end")
    accuracy = correct_count / length
    log_marker(f"{block_type}_block_{block_num}_accuracy_{accuracy:.2f}")

    # 6) Practice end message only
    if practice:
        end_txt = "Practice complete.\nPress any key to continue."
        end_stim = visual.TextStim(
            win, text=end_txt, color="white", wrapWidth=1.2, height=0.06
        )
        end_stim.draw(); win.flip()
        event.waitKeys(); check_for_quit(win)

    return accuracy

# ────────────────────────────────────────────────────────────────────────
#                               MAIN FUNCTION                            
# ────────────────────────────────────────────────────────────────────────

def main():
    global outlet
    # 1) Fullscreen dialog
    exp_info = {'Full screen? (yes/no)': 'yes'}
    dlg = gui.DlgFromDict(exp_info, title="Settings")
    if not dlg.OK:
        core.quit()
    fullscr = exp_info['Full screen? (yes/no)'].lower().startswith('y')

    # 2) Window & LSL Outlet
    win = visual.Window(fullscr=fullscr, color="black", units='height')
    info = StreamInfo(LSL_NAME, LSL_TYPE, 1, 0, 'string', LSL_ID)
    outlet = StreamOutlet(info)

    # 3) Welcome + 0-back baseline
    welcome = visual.TextStim(
        win,
        text="Welcome to N-back.\n\nFirst: 0-back baseline.\nPress any key to start.",
        color="white", wrapWidth=1.2, height=0.06
    )
    welcome.draw(); win.flip()
    event.waitKeys(); check_for_quit(win)
    run_block(win, outlet, n=0, length=BASELINE_TRIALS,
              practice=False, block_num=0)

    # 4) Practice loop
    practice_count = 0
    prompt = visual.TextStim(win, color="white", wrapWidth=1.2, height=0.06)

    prompt.text = "Would you like to practice?\n\nPress 'y' for yes and 'n' for no."
    prompt.draw(); win.flip()
    choice = event.waitKeys(keyList=['y', 'n', 'escape'])[0]
    if choice == 'escape':
        win.close(); core.quit()
    
    # UPDATE PRACTICE LIMITATION
    available = [1, 2, 3]
    if choice == 'y':
        while available:
            prompt.text = (
                 f"Which n-back to practice? ({'/'.join(str(x) for x in available)})"
            )
            prompt.draw(); win.flip()
            key = event.waitKeys(keyList=[str(x) for x in available])[0]
            n_prac = int(key)   # whichever key is pressed is then removed
            available.remove(n_prac)
            
            if key == 'escape':
                win.close(); core.quit()
            n_prac = int(key)

            practice_count += 1
            log_marker(f"practice_count_{practice_count}")
            run_block(win, outlet,
                      n=n_prac,
                      length=PRACTICE_TRIALS,
                      practice=True,
                      block_num=practice_count)
            
            if available:
                prompt.text = (
                "Would you like to practice again?\n\n"
                "Press 'y' for yes and 'n' for no."
                )
            prompt.draw(); win.flip()
            again = event.waitKeys(keyList=['y', 'n', 'escape'])[0]
            if again in ['n', 'escape']:
                break

    # 5) Start main blocks
    start_txt = visual.TextStim(
        win,
        text="Practice session over.\n\nPress any key to start main blocks.",
        color="white", wrapWidth=1.2, height=0.06
    )
    start_txt.draw(); win.flip()
    event.waitKeys(); check_for_quit(win)

    # 6) Main experiment with fixed breaks
    
    # UPDATE: Ensuring equal representation of blocks
    levels = [1, 2, 3]
    base = MAIN_BLOCKS // 3
    extra = MAIN_BLOCKS % 3
    order = levels * base + random.sample(levels, extra)
    random.shuffle(order)
    
    # main loop
    for mb, n_main in enumerate(order, start=1):
        run_block(win, outlet,
                  n=n_main,
                  length=MAIN_TRIALS,
                  practice=False,
                  block_num=mb)
        if mb < MAIN_BLOCKS:
            show_break(win, mb, MAIN_BLOCKS, BREAK_DURATION)

    # 7) Goodbye screen
    bye = visual.TextStim(
        win,
        text="All done! Thank you.\nPress any key to exit.",
        color="white", wrapWidth=1.2, height=0.06
    )
    bye.draw(); win.flip()
    event.waitKeys()

    # ─────────────────────────────────────────────────────────────────────
    # Save redundancy CSV with all markers & timestamps
    with open("marker_log.csv", mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["marker", "timestamp"])
        writer.writerows(marker_log)

    # Clean up
    win.close()
    core.quit()

if __name__ == "__main__":
    main()