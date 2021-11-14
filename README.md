# Exploration of NHL's API to extract NHL play event and aggregated stats for players.
Main idea was to test a simle Expected Goal(XG) based on logistic a model. Based on this then see which players ended up being the best based on these XG values. Finally compare Connor McDavid's XG values between other players and compare his XG values season per season during his NHL career.

However, the XG model doesnt seem to be doing that great as the model doesnt take into accoun additional play context such as where was the puck previous to the shot, and when, and where were other players, etc...

The XG model seem to conceptually making sense based on how the XG values varies depending on where the shot is taking place in relation to the distance and angle to the goal.
