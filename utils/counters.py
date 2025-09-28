# Pushup counter
def pushup_counter(angle, counter, stage):
    if angle > 160:   # arms extended
        stage = "up"
    if angle < 70 and stage == "up":
        stage = "down"
        counter += 1
    return counter, stage

# Squat counter
def squat_counter(angle, counter, stage):
    if angle > 160:
        stage = "up"
    if angle < 90 and stage == "up":
        stage = "down"
        counter += 1
    return counter, stage

# Jump counter (based on vertical movement)
def jump_counter(y_position, baseline, counter, stage):
    if y_position > baseline + 0.05:
        stage = "up"
    if y_position < baseline - 0.05 and stage == "up":
        stage = "down"
        counter += 1
    return counter, stage
