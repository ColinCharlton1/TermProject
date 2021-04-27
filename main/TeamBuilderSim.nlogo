globals [
  process-id
  ; environment related globals
  rock-adjustment
  tree-adjustment
  bush-adjustment
  puddle-adjustment
  river-toggle
  stone-centering
  stone-density
  stockpile-num
  max-eating
  tree-reproduce-ticks
  bush-reproduce-ticks
  max-trees
  max-bushes
  base-berry-num
  base-wood-num
  max-ticks
  sorted-patches
  world-updates
  actor-stats
  generation-num
  save-directory

  ; actor related globals
  pop-mod
  team1-pop-dist
  team2-pop-dist
  max-hunger
  start-hunger
  max-carried-wood
  max-carried-stone
  red-advantage
  purple-advantage
  red-bonus
  purple-bonus
  team-seperation
  team-split-style

  ; h = hunger
  move-hcost
  water-move-hcost
  cut-wood-hcost
  mine-hcost
  build-wood-hcost
  build-stone-hcost
  build-bridge-hcost
  destroy-wood-hcost
  destroy-stone-hcost
  destroy-stock-hcost

  ; reward related globals
  death-penalty
  eating_r
  cut_tree_r
  mine_r
  build_wood_r
  build_stone_r
  build_bridge_r
  destroy_enemy_wood_r
  destroy_enemy_stone_r
  destroy_enemy_stockpile_r
  destroy_friendly_r
  repeat_turn_penalty

  wood__water_b
  wood__f_stock_b
  wood__e_stock_b
  stone__water_b
  stone__f_stock_b
  stone__e_stock_b
  all_structs__center_world_b
  last_struct_built_b
  destroy_e__e_struct_b
  destroy_e__f_struct_b
  destroy_e__e_stock_b
  destroy_e__f_stock_b

  wood__f_adj_b
  wood__e_adj_b
  stone__f_adj_b
  stone__e_adj_b
  bridge__water_adj_b
  bridge__bridge_adj_b
]

breed [halos halo]
breed [trees tree]
breed [bushes bush]
breed [actors actor]

actors-own [
  death-mark
  team-color
  hunger
  carried-wood
  carried-stone
  reward
  cumulative_reward
  turn-restriction
  last-build-position
  ; int with meaning as shown in execute-actions
  action
  death-action
  ; below is all just for tracking stats to analyze in python after generation ends
  times-eaten
  trees-harvested
  stone-mined
  wood-built
  stone-built
  bridges-built
  team-wood-destroyed
  team-stone-destroyed
  enemy-wood-destroyed
  enemy-stone-destroyed
]

bushes-own [
  berries
]

trees-own [
  wood
]

patches-own [
  patch-veg-score
  patch-veg-num
  patch-actor-score
  patch-actor-num
  boxedc
]

;; *********************************** Functions For Sending Data To Python ***********************************

to-report env-vegetation
  report map [p -> [veg-total] of p] sorted-patches
end

to-report env-actors
  report map [p -> [actor-total] of p] sorted-patches
end

to-report env-types
  report map [p -> [boxedc] of p] sorted-patches
end

to-report get-updates
  if (length world-updates = 0) [report 0]
  report world-updates
end

to-report get-actor-data
  set world-updates (list)
  if (count actors with [death-mark != 3] = 0) [ report 0 ]
  report map [a -> [(list pxcor pycor (safe-patch-ahead) (heading-box heading) (hunger / max-hunger) (carried-wood / max-carried-wood) (carried-stone / max-carried-stone) my-advantage action reward)] of a] (sort actors with [death-mark != 3])
end

to-report safe-patch-ahead
  if (patch-ahead 1 = nobody) [report 0]
  report ([boxedc] of patch-ahead 1) / 25
end

to-report get-ids
  if (count actors with [death-mark != 3] = 0) [ report 0 ]
  let partition (list (length team1-pop-dist + length team2-pop-dist + 1))
  let place 0
  let last-place 0
  (foreach (sentence team1-pop-dist team2-pop-dist) [
    [species-members] ->
    set last-place place
    set place place + species-members
    set partition lput (count actors with [who >= last-place and who < place and death-mark != 3]) partition
  ])
  report (sentence partition (map [a -> [who] of a] (sort actors with [death-mark != 3])))
end

to-report get-actor-stats
  ask actors [
    set actor-stats lput
    (list
      who
      ticks
      times-eaten
      trees-harvested
      stone-mined
      wood-built
      stone-built
      bridges-built
      team-wood-destroyed
      team-stone-destroyed
      enemy-wood-destroyed
      enemy-stone-destroyed) actor-stats
  ]
  set actor-stats sort-by [ [l1 l2] -> item 0 l1 < item 0 l2] actor-stats
  export-view (word save-directory "island_" process-id "_gen_" generation-num ".png")
  set generation-num generation-num + 1
  report actor-stats
end

to-report get-stockpiles
  let red-stockpiles (count patches with [pcolor = red])
  let purple-stockpiles (count patches with [pcolor = violet])
  report list red-stockpiles purple-stockpiles
end

;; *********************************** Data Helper Functions ***********************************

to-report my-advantage
  if (team-color = red) [report red-advantage]
  report purple-advantage
end

to-report heading-box [h]
  if (h < 45) [report 0.8]
  if (h < 135) [report 0.6]
  if (h < 225) [report 0.2]
  if (h < 315) [report 0.4]
  report 1
end

to-report color-box [c]
  if (c < 5.3) [report 10] ; more full rock patches
  if (c < 6) [report 9] ; emptier rock patches
  if (c < 15) [report 13] ; red stone
  if (c = 15) [report 16] ; red stockpile
  if (c < 20) [report 12] ; red wood
  if (c <= brown + 1) [report 3] ; bridges
  if (c < 60) [report 1] ; grass patches
  if (c < 100) [report 5] ; water patches
  if (c < 115) [report 21] ; violet stone
  if (c = 115) [report 24] ; violet stockpile
  report 20 ; violet wood
end

to-report veg-total ; turtle/patch procedure
  if (patch-veg-num = 0) [report 0]
  report round (patch-veg-score / patch-veg-num)
end

to-report actor-total ; turtle/patch procedure
  if (patch-actor-num = 0) [report 0]
  report round (patch-actor-score / patch-actor-num)
end

;; *********************************** Functions For Starting Generation From Python ***********************************

to set-env-params [ rock-adj tree-adj bush-adj pud-adj river-tog ww wh ps max-t berry-start wood-start stone-dense stone-cent
                    start-pop stock-num t-seperation t-split-style start-hgr max-eat max-hgr pro-id run-name]
  set rock-adjustment rock-adj
  set tree-adjustment tree-adj
  set bush-adjustment bush-adj
  set puddle-adjustment pud-adj
  set river-toggle river-tog
  set stone-centering stone-cent
  set stone-density stone-dense
  set base-berry-num berry-start
  set base-wood-num wood-start
  set stockpile-num stock-num
  set team-seperation t-seperation
  set team-split-style t-split-style
  set start-hunger start-hgr
  set max-eating max-eat
  set max-hunger max-hgr
  set pop-mod start-pop
  set max-carried-wood 750
  set max-carried-stone 250
  set max-trees 8000
  set max-bushes 8000
  set tree-reproduce-ticks 125
  set bush-reproduce-ticks 75
  resize-world 0 (ww - 1) 0 (wh - 1)
  set-patch-size ps
  set max-ticks max-t
  set process-id pro-id
  set save-directory run-name
end

; this is an awful function, but its arguably the best way to send over the data from python
to set-reward-params [death eat_r cut_r mne_r bld_w_r bld_s_r bld_b_r dest_ew_r dest_es_r dest_estock_r dest_f
                      w_wat_b w_f_stock_b w_e_stock_b s_wat_b s_f_stock_b s_e_stock_b world_center_b last_struct_b
                      dest_e_e_struct dest_e_f_struct dest_e_e_stock dest_e_f_stock
                      w_f_adj w_e_adj s_f_adj s_e_adj br_w_adj br_br_adj rpt_turn_penalty
                      mv-cost wmv-cost cut-cost mne-cost bld-w-cost bld-s-cost bld-b-cost
                      dest-w-cost dest-s-cost dest-stock-cost]

  set death-penalty death
  set eating_r eat_r
  set cut_tree_r cut_r
  set mine_r mne_r
  set build_wood_r bld_w_r
  set build_stone_r bld_s_r
  set build_bridge_r bld_b_r
  set destroy_enemy_wood_r dest_ew_r
  set destroy_enemy_stone_r dest_es_r
  set destroy_enemy_stockpile_r dest_estock_r
  set destroy_friendly_r dest_f

  set wood__water_b w_wat_b
  set wood__f_stock_b w_f_stock_b
  set wood__e_stock_b w_e_stock_b
  set stone__water_b s_wat_b
  set stone__f_stock_b s_f_stock_b
  set stone__e_stock_b s_e_stock_b
  set all_structs__center_world_b world_center_b
  set last_struct_built_b last_struct_b

  set destroy_e__e_struct_b dest_e_e_struct
  set destroy_e__f_struct_b dest_e_f_struct
  set destroy_e__e_stock_b dest_e_e_stock
  set destroy_e__f_stock_b dest_e_f_stock

  set wood__f_adj_b w_f_adj
  set wood__e_adj_b w_e_adj
  set stone__f_adj_b s_f_adj
  set stone__e_adj_b s_e_adj
  set bridge__water_adj_b br_br_adj
  set bridge__bridge_adj_b br_br_adj

  set move-hcost mv-cost
  set water-move-hcost wmv-cost
  set cut-wood-hcost cut-cost
  set mine-hcost mne-cost
  set build-wood-hcost bld-w-cost
  set build-stone-hcost bld-s-cost
  set build-bridge-hcost bld-b-cost
  set destroy-wood-hcost dest-w-cost
  set destroy-stone-hcost dest-s-cost
  set destroy-stock-hcost dest-stock-cost
end


to setup [reddist purpdist]
  clear-generation
  set team1-pop-dist reddist
  set team2-pop-dist purpdist
  set world-updates (list)
  set actor-stats (list)
  set-default-shape trees "tree"
  set-default-shape bushes "bush"
  set-default-shape halos "thin square"
  setup-environment
  reset-ticks
end

to setup-environment
  setup-patches
  setup-species
  setup-bushes
  setup-trees
  ask patches [
    set patch-veg-score 25 * count trees-here + 10 * count bushes-here
    set patch-veg-num count bushes-here + count trees-here
    set patch-actor-score 10 * count actors-here with [team-color = red] + 25 * count actors-here with [team-color = violet]
    set patch-actor-num count actors-here
    set boxedc color-box pcolor
  ]
  set sorted-patches sort patches
  create-halos 1 [setxy round (max-pxcor / 2) round (max-pycor / 2) set color yellow set shape "thin square" facexy pxcor pycor + 1 set size 1.5]
end

to clear-generation
  clear-all-plots
  clear-output
  clear-patches
  clear-ticks
  clear-turtles
end

;; *********************************** Main Step Functions ***********************************

to end-step
  set-team-bonuses
  execute-actions
  if (ticks != 0) [
    if ( remainder ticks tree-reproduce-ticks = 0) [reproduce-trees]
    if ( remainder ticks bush-reproduce-ticks = 0) [reproduce-bushes]
  ]
  kill-actors
  tick
end

to kill-actors
  ask actors with [death-mark = 3]
  [
    ifelse (team-color = red) [set patch-actor-score patch-actor-score - 10]
    [set patch-actor-score patch-actor-score - 25]
    set patch-actor-num patch-actor-num - 1
    set world-updates lput (list pxcor pycor 1 actor-total) world-updates
    set actor-stats lput
    (list
      who
      ticks
      times-eaten
      trees-harvested
      stone-mined
      wood-built
      stone-built
      bridges-built
      team-wood-destroyed
      team-stone-destroyed
      enemy-wood-destroyed
      enemy-stone-destroyed) actor-stats

    die
  ]
end

to set-team-bonuses
  ; teams have a small buffer from their stockpiles, should never usually be 0
  let red-structs (count patches with [shade-of? red pcolor])
  let red-stockpiles 5 * (count patches with [pcolor = red])
  let red-pop (count actors with [team-color = red])
  let red-score max list 1 ( ( red-stockpiles + red-structs + red-pop ))

  let purple-structs (count patches with [shade-of? violet pcolor])
  let purple-stockpiles 5 * (count patches with [pcolor = violet])
  let purple-pop (count actors with [team-color = violet])
  let purple-score max list 1 ( ( purple-stockpiles + purple-structs + purple-pop ))

  set red-advantage (red-score /    (red-score + purple-score))
  set purple-advantage (purple-score / (red-score + purple-score))

  set red-bonus    red-advantage    * max list 0.5 (min list 1.5 (pop-mod / sum team1-pop-dist) )
  set purple-bonus purple-advantage * max list 0.5 (min list 1.5 (pop-mod / sum team2-pop-dist) )
end

;; *********************************** Actor Setup Functions ***********************************

to setup-species
  (ifelse (team-split-style = 0) [ setup-species-style-0 ]
  (team-split-style = 1) [ setup-species-style-1 ]
  (team-split-style = 2) [ setup-species-style-2 ])
end

to setup-species-style-0
  let team1-colors (list 15 25 13.5 16.5 27)
  let team2-colors (list 116.8 117.8 114.5 85 95)
  (foreach range (length team1-pop-dist) [
    [index] ->
    create-actors (item index team1-pop-dist) [
      ; red team (team 1)
      set team-color red
      setxy round (random (max-pxcor * team-seperation)) round random-pycor
      set color item index team1-colors
      set hunger start-hunger
      while [count actors-here > 1 or not shade-of? green pcolor] [setxy round (random (max-pxcor * team-seperation)) round random-pycor]
      facexy pxcor - 1 pycor
    ]
  ])
  (foreach range (length team2-pop-dist) [
    [index] ->
    create-actors (item index team2-pop-dist) [
      ; purple team  (team 2)
      set team-color violet
      set shape "default-plus"
      setxy round ((random (max-pxcor * team-seperation)) + max-pxcor * (1 - team-seperation)) round random-pycor
      set color item index team2-colors
      set hunger start-hunger
      while [count actors-here > 1 or not shade-of? green pcolor] [setxy round ((random (max-pxcor * team-seperation)) + max-pxcor * (1 - team-seperation)) round random-pycor]
      facexy pxcor + 1 pycor
    ]
  ])
end

to setup-species-style-1
  let team1-colors (list 15 25 13.5 16.5 27)
  let team2-colors (list 116.8 117.8 114.5 85 95)
  (foreach range (length team1-pop-dist) [
    [index] ->
    create-actors (item index team1-pop-dist) [
      ; red team (team 1)
      set team-color red
      setxy round random-pxcor round (random (max-pycor * team-seperation))
      set color item index team1-colors
      set hunger start-hunger
      while [count actors-here > 1 or not shade-of? green pcolor] [setxy round random-pxcor round (random (max-pycor * team-seperation))]
      facexy pxcor - 1 pycor
    ]
  ])
  (foreach range (length team2-pop-dist) [
    [index] ->
    create-actors (item index team2-pop-dist) [
      ; purple team  (team 2)
      set team-color violet
      set shape "default-plus"
      setxy round random-pxcor round (random (max-pycor * team-seperation) + max-pycor * (1 - team-seperation))
      set color item index team2-colors
      set hunger start-hunger
      while [count actors-here > 1 or not shade-of? green pcolor] [setxy round random-pxcor round (random (max-pycor * team-seperation) + max-pycor * (1 - team-seperation))]
      facexy pxcor + 1 pycor
    ]
  ])
end

to setup-species-style-2
  let team1-colors (list 15 25 13.5 16.5 27)
  let team2-colors (list 116.8 117.8 114.5 85 95)
  let max-percent max list 0.1 (team-seperation)
  let max-total (max-pxcor + max-pycor) * max-percent
  let min-total (max-pxcor + max-pycor) * (1 - max-percent)
  (foreach range (length team1-pop-dist) [
    [index] ->
    create-actors (item index team1-pop-dist) [
      ; red team (team 1)
      set team-color red
      let randyc round random-pxcor
      let randxc round random-pycor
      while [(not shade-of? green [pcolor] of patch randxc randyc) or
            (count actors-here > 1) or
            (randyc + randxc >= max-total)]
      [set randyc round random-ycor set randxc round random-xcor]
      setxy randxc randyc
      set color item index team1-colors
      set hunger start-hunger
      facexy pxcor - 1 pycor
    ]
  ])
  (foreach range (length team2-pop-dist) [
    [index] ->
    create-actors (item index team2-pop-dist) [
      ; purple team  (team 2)
      set team-color violet
      set shape "default-plus"
      let randyc round random-pxcor
      let randxc round random-pycor
      while [(not shade-of? green [pcolor] of patch randxc randyc) or
            (count actors-here > 1) or
            (randyc + randxc <= min-total)]
      [set randyc round random-ycor set randxc round random-xcor]
      setxy randxc randyc
      set color item index team2-colors
      set hunger start-hunger
      facexy pxcor + 1 pycor
    ]
  ])
end

;; *********************************** Actor Action Functions ***********************************

to execute-actions
  ask actors with [death-mark > 0] [ set death-mark death-mark + 1 set action -2]
  ask actors with [death-mark = 0] [

    ;penalize actors for becoming addicted to spinning
    ifelse (action != 4 and action != 5) [set reward 0 set turn-restriction 0]
    [
      set reward repeat_turn_penalty * turn-restriction
      set turn-restriction 1
      ifelse (pcolor = 97) [set hunger hunger - water-move-hcost]
      [set hunger hunger - move-hcost]
    ]

    (ifelse
    (action = 0) [ move 0 1 ]
    (action = 1) [ move -1 0 ]
    (action = 2) [ move 0 -1 ]
    (action = 3) [ move 1 0 ]
    (action = 4) [ rt 90 ]
    (action = 5) [ lt 90 ]
    (action = 6) [eat-berries]
    (action = 7) [cut-tree]
    (action = 8) [mine-stone]
    (action = 9) [build-wooden-structure]
    (action = 10) [build-stone-structure]
    (action = 11) [build-bridge]
    (action = 12) [destroy-structure])

    set hunger hunger - 1 ; cost of living
    if (hunger <= 0) [set death-mark 1 set action -2 set reward death-penalty * ( 1 - (ticks / max-ticks) )]

    (
    ifelse
    (team-color = red and reward >= 0) [set reward min list 1 (reward * red-bonus)]
    (team-color = violet and reward >= 0) [set reward min list 1 (reward * purple-bonus)]
    (team-color = red) [set reward reward * purple-bonus]
    (team-color = violet) [set reward reward * red-bonus]
    )
    set cumulative_reward cumulative_reward + reward
  ]
end


to move [x y]
  facexy (pxcor + x) (pycor + y)
  let target-patch patch-ahead 1
  if (target-patch != nobody)
  [
    let patch-color [pcolor] of target-patch
    if (shade-of? green patch-color or patch-color = 97 or patch-color = brown) [
      ifelse (team-color = red)
      [
        set patch-actor-score patch-actor-score - 10
        set world-updates lput (list pxcor pycor 1 actor-total) world-updates
        fd 1
        set patch-actor-score patch-actor-score + 10
        set world-updates lput (list pxcor pycor 1 actor-total) world-updates
      ]
      [
        set patch-actor-score patch-actor-score - 25
        set world-updates lput (list pxcor pycor 1 actor-total) world-updates
        fd 1
        set patch-actor-score patch-actor-score + 25
        set world-updates lput (list pxcor pycor 1 actor-total) world-updates
      ]
      set reward 0.0001
    ]
  ]
  ifelse (pcolor = 97) [set hunger hunger - water-move-hcost] ; much more effort to move through water
  [set hunger hunger - move-hcost]
end

to eat-berries ; turtle procedure
  if (count bushes-here > 0)
  [
    let eat-limit min list (max-hunger - hunger) round ( max-eating / 2 + random (max-eating / 2) )
    ifelse ( eat-limit < round (max-eating / 2) )
    [ set hunger hunger - round (max-eating / 4) set reward eating_r / -5] ; throw up from trying to eat too much
    [
      ask max-one-of bushes-here [berries]
      [
        set eat-limit min (list eat-limit berries)
        set berries berries - eat-limit
        if (berries = 0) [
          set patch-veg-score patch-veg-score - 10
          set patch-veg-num patch-veg-num - 1
          set world-updates lput (list pxcor pycor 0 veg-total) world-updates
          die
        ]
      ]
      set reward eating_r
      set times-eaten times-eaten + 1
      set hunger hunger + eat-limit
    ]
  ]
end

to cut-tree ; turtle procedure
  let t-here trees-here
  ifelse (count t-here > 0 and hunger >= cut-wood-hcost)
  [
    let harvested-wood 0
    ask one-of t-here [ set harvested-wood wood die ]
    set patch-veg-score patch-veg-score - 25
    set patch-veg-num patch-veg-num - 1
    set world-updates lput (list pxcor pycor 0 veg-total) world-updates
    set trees-harvested trees-harvested + 1
    set carried-wood min list max-carried-wood (carried-wood + harvested-wood)
    set hunger hunger - cut-wood-hcost
    set reward cut_tree_r
  ]
  [set hunger hunger - 1] ; wasting energy
end

to mine-stone ; turtle procedure
  ifelse (patch-ahead 1 != nobody and hunger >= mine-hcost)
  [
    ifelse ([pcolor] of patch-ahead 1 < 10)
    [
      ask patch-ahead 1
      [
        ifelse (pcolor + 0.3 >= 5.9)
        [
          set pcolor 57
          set boxedc color-box pcolor
          set world-updates lput (list pxcor pycor 2 boxedc) world-updates
        ]
        [
          set pcolor pcolor + 0.3
          set boxedc color-box pcolor
          set world-updates lput (list pxcor pycor 2 boxedc) world-updates
        ]
      ]
      set stone-mined stone-mined + 1
      set carried-stone min list max-carried-stone (carried-stone + stone-density + (random (stone-density / 2)))
      set hunger hunger - mine-hcost
      set reward mine_r
    ]
    [set hunger hunger - 1] ; wasting energy
  ]
  [set hunger hunger - 1] ; wasting energy
end

to build-wooden-structure ; turtle procedure
  ifelse (patch-ahead 1 != nobody and hunger >= build-wood-hcost)
  [
    ifelse (carried-wood >= 50 and shade-of? green ([pcolor] of patch-ahead 1) and count [actors-here] of patch-ahead 1 = 0)
    [
      let my-color team-color
      reward-wood-build my-color
      ask patch-ahead 1 [
        set pcolor my-color + 2.5
        set boxedc color-box pcolor
        ask trees-here [die]
        ask bushes-here [die]
        set patch-veg-score 0
        set patch-veg-num 0
        set world-updates lput (list pxcor pycor 0 0) world-updates
        set world-updates lput (list pxcor pycor 2 boxedc) world-updates
      ]
      set last-build-position (list pxcor pycor)
      set wood-built wood-built + 1
      set hunger hunger - build-wood-hcost
      set carried-wood carried-wood - 50
    ]
    [set hunger hunger - 1] ; wasting energy
  ]
  [set hunger hunger - 1] ; wasting energy
end

to build-stone-structure ; turtle procedure
  ifelse (patch-ahead 1 != nobody and hunger >= build-stone-hcost)
  [
    ifelse (carried-stone >= 50 and
           shade-of? green ([pcolor] of patch-ahead 1) and
           count [actors-here] of patch-ahead 1 = 0)
    [
      let my-color team-color
      reward-stone-build my-color
      ask patch-ahead 1 [
        set pcolor my-color - 2.5
        set boxedc color-box pcolor
        ask trees-here [die]
        ask bushes-here [die]
        set patch-veg-score 0
        set patch-veg-num 0
        set world-updates lput (list pxcor pycor 0 0) world-updates
        set world-updates lput (list pxcor pycor 2 boxedc) world-updates
      ]
      set last-build-position (list pxcor pycor)
      set stone-built stone-built + 1
      set hunger hunger - build-stone-hcost
      set carried-stone carried-stone - 50
    ]
    [set hunger hunger - 1] ; wasting energy
  ]
  [set hunger hunger - 1] ; wasting energy
end

to build-bridge ; turtle procedure
  ifelse (patch-ahead 1 != nobody and hunger >= build-bridge-hcost)
  [
    ifelse (carried-stone >= 10 and
           carried-wood >= 30 and
           ([pcolor] of patch-ahead 1) = 97 and
           ([count neighbors4 with [shade-of? green pcolor or pcolor = brown]] of patch-ahead 1) > 0)
    [
      reward-bridge-build
      ask patch-ahead 1 [
        set pcolor brown
        set boxedc color-box brown
        set world-updates lput (list pxcor pycor 2 boxedc) world-updates
      ]
      set last-build-position (list pxcor pycor)
      set bridges-built bridges-built + 1
      set hunger hunger - build-bridge-hcost
      set carried-stone carried-stone - 10
      set carried-wood carried-wood - 30
    ]
    [set hunger hunger - 1] ; wasting energy
  ]
  [set hunger hunger - 1] ; wasting energy
end

to destroy-structure ; turtle procedure
  ifelse (patch-ahead 1 != nobody)
  [
    let destroy-color [pcolor] of patch-ahead 1
    let my-color team-color
    let enemy-color red
    if (team-color = red) [set enemy-color violet]
    (ifelse (destroy-color = enemy-color) [destroy-stockpile]
    (destroy-color = enemy-color - 2.5) [destroy-stone enemy-color destroy-color]
    (destroy-color = enemy-color + 2.5) [destroy-wood enemy-color destroy-color]
    (count neighbors with [shade-of? my-color pcolor] + (8 - count neighbors) > 3)
    [ (ifelse (destroy-color = my-color - 2.5) [destroy-stone enemy-color destroy-color]
      (destroy-color = my-color + 2.5) [destroy-wood enemy-color destroy-color]) ]
    [ set hunger hunger - 1 ]) ; wasting energy
  ]
  [set hunger hunger - 1] ; wasting energy
end

to destroy-stockpile
  ifelse (hunger >= destroy-stock-hcost)
  [
    reward-destroy-stock
    set hunger hunger - destroy-stock-hcost
    ask patch-ahead 1
    [
      set pcolor 57
      set boxedc color-box green
      set world-updates lput (list pxcor pycor 2 boxedc) world-updates
    ]
  ]
  [set hunger hunger - 1] ; wasting energy
end

to destroy-wood [ec dc]
  ifelse (hunger >= destroy-wood-hcost)
  [
    set carried-wood min list max-carried-wood (carried-wood + 5 + random 5)
    reward-destroy-wood ec dc
    set hunger hunger - destroy-wood-hcost
    ifelse (shade-of? ec dc) [set enemy-wood-destroyed enemy-wood-destroyed + 1]
    [set team-wood-destroyed team-wood-destroyed + 1]
    ask patch-ahead 1
    [
      set pcolor 57
      set boxedc color-box green
      set world-updates lput (list pxcor pycor 2 boxedc) world-updates
    ]
  ]
  [set hunger hunger - 1] ; wasting energy
end

to destroy-stone [ec dc]
  ifelse (hunger >= destroy-stone-hcost)
  [
    set carried-stone min list max-carried-stone (carried-stone + 5 + random 5)
    reward-destroy-stone ec dc
    set hunger hunger - destroy-stone-hcost
    ifelse (shade-of? ec dc) [set enemy-stone-destroyed enemy-stone-destroyed + 1]
    [set team-stone-destroyed team-stone-destroyed + 1]
    ask patch-ahead 1
    [
      set pcolor 57
      set boxedc color-box green
      set world-updates lput (list pxcor pycor 2 boxedc) world-updates
    ]
  ]
  [set hunger hunger - 1] ; wasting energy
end

;; *********************************** Reward Functions ***********************************

to reward-wood-build [my-color]
  let bonus 0

  let enemy-color red
  if (team-color = red) [set enemy-color violet]

  ask patch-ahead 1 [
    set bonus bonus + wood__f_adj_b * min list 5 (count neighbors with [shade-of? my-color pcolor])
    set bonus bonus + wood__e_adj_b * min list 5 (count neighbors with [shade-of? enemy-color pcolor])
    set bonus bonus + all_structs__center_world_b / max list 1 (distance patch round (max-pxcor / 2) round (max-pycor / 2))

    if (count patches with [shade-of? blue pcolor] > 0)
    [set bonus bonus + wood__water_b / distance (min-one-of other (patches with [shade-of? blue pcolor]) [distance myself])]

    if (count patches with [pcolor = my-color] > 0)
    [set bonus bonus + wood__f_stock_b / distance (min-one-of other (patches with [pcolor = my-color]) [distance myself])]

    if (count patches with [pcolor = enemy-color] > 0)
    [set bonus bonus + wood__e_stock_b / distance (min-one-of other (patches with [pcolor = enemy-color]) [distance myself])]
    if ([last-build-position] of myself != 0)
    [
      let lastx [item 0 last-build-position] of myself
      let lasty [item 1 last-build-position] of myself
      set bonus bonus + last_struct_built_b / max list 1 distance (patch lastx lasty)
    ]
  ]
  set bonus bonus + min list 1 (0.02 * wood-built)
  ifelse (bonus > 1) [ set reward (build_wood_r * bonus) ] ; good placement reward
  [set reward build_wood_r] ; bad placement rewards
end

to reward-stone-build [my-color]
  let bonus 0

  let enemy-color red
  if (team-color = red) [set enemy-color violet]

  ask patch-ahead 1 [
    set bonus bonus + stone__f_adj_b * min list 5 (count neighbors with [shade-of? my-color pcolor])
    set bonus bonus + stone__e_adj_b * min list 5 (count neighbors with [shade-of? enemy-color pcolor])
    set bonus bonus + all_structs__center_world_b / max list 1 (distance patch round (max-pxcor / 2) round (max-pycor / 2))

    if (count patches with [shade-of? blue pcolor] > 0)
      [set bonus bonus + stone__water_b / distance (min-one-of other (patches with [shade-of? blue pcolor]) [distance myself])]

    if (count patches with [pcolor = my-color] > 0)
      [set bonus bonus + stone__f_stock_b / max list 1 distance (min-one-of other (patches with [pcolor = my-color]) [distance myself])]

    if (count patches with [pcolor = enemy-color] > 0)
      [set bonus bonus + stone__e_stock_b / distance (min-one-of other (patches with [pcolor = enemy-color]) [distance myself])]
    if ([last-build-position] of myself != 0)
    [
      let lastx [item 0 last-build-position] of myself
      let lasty [item 1 last-build-position] of myself
      set bonus bonus + last_struct_built_b / max list 1 distance (patch lastx lasty)
    ]
  ]
  set bonus bonus + min list 0.5 (0.02 * stone-built)
  ifelse (bonus > 1) [ set reward (build_stone_r * bonus) ] ; good placement reward
  [set reward build_stone_r] ; bad placement reward
end

to reward-bridge-build
  let bonus 0

  ask patch-ahead 1 [
    set bonus bonus + bridge__water_adj_b * min list 5 (count neighbors with [shade-of? blue pcolor])
    set bonus bonus + bridge__bridge_adj_b * min list 5 (count neighbors with [shade-of? brown pcolor])
    set bonus bonus + all_structs__center_world_b / max list 1 distance patch round (max-pxcor / 2) round (max-pycor / 2)
  ]
  set bonus bonus + min list 0.5 (0.05 * bridges-built)
  ifelse (bonus > 1) [ set reward (build_bridge_r * bonus) ] ; good placement reward
  [set reward build_bridge_r] ; bad placement reward
end

to reward-destroy-wood [ec dc]
  let bonus 0
  let my-color team-color
  ask patch-ahead 1
  [
    if (count other patches with [pcolor = ec] > 0)
    [set bonus bonus + destroy_e__e_stock_b / distance (min-one-of other (patches with [pcolor = ec]) [distance myself])]
    if (count other patches with [pcolor = my-color] > 0)
    [set bonus bonus + destroy_e__f_stock_b / distance (min-one-of other (patches with [pcolor = my-color]) [distance myself])]
    if (count other patches with [shade-of? ec pcolor] > 0)
    [set bonus bonus + destroy_e__e_struct_b / distance (min-one-of other (patches with [shade-of? ec pcolor]) [distance myself])]
    if (count other patches with [shade-of? my-color pcolor] > 0)
    [set bonus bonus + destroy_e__f_struct_b / distance (min-one-of other (patches with [shade-of? my-color pcolor]) [distance myself])]
  ]
  set bonus bonus + max list -1 (min list 2 (0.08 * (enemy-wood-destroyed + enemy-stone-destroyed - team-wood-destroyed - team-stone-destroyed)))
  ifelse (shade-of? my-color dc) [set reward destroy_friendly_r]
  [set reward destroy_enemy_wood_r * bonus]
end

to reward-destroy-stone [ec dc]
  let bonus 0
  let my-color team-color
  ask patch-ahead 1
  [
    if (count other patches with [pcolor = ec] > 0)
    [set bonus bonus + destroy_e__e_stock_b / distance (min-one-of other (patches with [pcolor = ec]) [distance myself])]
    if (count other patches with [pcolor = my-color] > 0)
    [set bonus bonus + destroy_e__f_stock_b / distance (min-one-of other (patches with [pcolor = my-color]) [distance myself])]
    if (count other patches with [shade-of? ec pcolor] > 0)
    [set bonus bonus + destroy_e__e_struct_b / distance (min-one-of other (patches with [shade-of? ec pcolor]) [distance myself])]
    if (count other patches with [shade-of? my-color pcolor] > 0)
    [set bonus bonus + destroy_e__f_struct_b / distance (min-one-of other (patches with [shade-of? my-color pcolor]) [distance myself])]
  ]
  set bonus bonus + max list -1 (min list 2 (0.08 * (enemy-wood-destroyed + enemy-stone-destroyed - team-wood-destroyed - team-stone-destroyed)))
  ifelse (shade-of? my-color dc) [set reward destroy_friendly_r]
  [set reward destroy_enemy_stone_r * bonus]
end

to reward-destroy-stock
  set reward destroy_enemy_stockpile_r
end

;; *********************************** Environment Update Functions ***********************************

to reproduce-trees
  let newcors (list)
  ask trees [
    if (random 30 = 0 and count trees < max-trees) [
       set newcors lput (list pxcor pycor) newcors
    ]
  ]
  let index 0
  create-trees min (list (length newcors) (max-trees - count trees)) [
    setxy spreadcor (item 0 item index newcors) 5 true spreadcor (item 1 item index newcors) 5 false
    set wood base-wood-num + random (base-wood-num / 2)
    set size 2
    set color (list 10 (160 + random 40) 10 150)
    ifelse ((not shade-of? green pcolor) or
             tree-density > 5)
    [die]
    [
      set patch-veg-score patch-veg-score + 25
      set patch-veg-num patch-veg-num + 1
      set world-updates lput (list pxcor pycor 0 veg-total) world-updates
    ]
    set index index + 1
  ]
end

to reproduce-bushes
  let newcors (list)
  ask bushes [
    if (random 15 = 0 and count bushes < max-bushes) [
       set newcors lput (list pxcor pycor) newcors
    ]
    set berries base-berry-num + random (base-berry-num / 2)
  ]
  let index 0
  create-bushes min (list (length newcors) (max-bushes - count bushes)) [
    setxy spreadcor (item 0 item index newcors) 4 true spreadcor (item 1 item index newcors) 4 false
    set size 1.5
    facexy xcor 0
    set color (list 30 (140 + random 40) 80 75)
    set berries base-berry-num + random (base-berry-num / 2)
    ifelse ((not shade-of? green pcolor) or
             bush-density > 5)
    [die]
    [
      set patch-veg-score patch-veg-score + 10
      set patch-veg-num patch-veg-num + 1
      set world-updates lput (list pxcor pycor 0 veg-total) world-updates
    ]
    set index index + 1
   ]
end

;; *********************************** Vegitation Setup ***********************************

to setup-trees
  while [count trees < (count patches with [shade-of? green pcolor] * (0.1 + 0.01 * tree-adjustment))] [
    let randxc random-xcor
    let randyc random-ycor
    while [1 < [tree-density] of patch randxc randyc] [set randxc random-xcor set randyc random-ycor]
    create-trees 8 [
      setxy spreadcor randxc 8 true spreadcor randyc 8 false
      set wood base-wood-num + random (base-wood-num / 2)
      set size 2
      set color (list 10 (160 + random 40) 10 150)
      if ((not shade-of? green pcolor) or
          tree-density > 5)
      [die]
    ]
  ]
end

to setup-bushes
  while [count bushes < (count patches with [shade-of? green pcolor] * (0.1 + 0.01 * bush-adjustment))] [
    let randxc random-xcor
    let randyc random-ycor
    while [1 < [bush-density] of patch randxc randyc] [set randxc random-xcor set randyc random-ycor]
    create-bushes 12 [
      setxy spreadcor randxc 8 true spreadcor randyc 8 false
      set size 1.5
      facexy xcor 0
      set color (list 30 (140 + random 40) 80 75)
      set berries base-berry-num + random (base-berry-num / 2)
      if ((not shade-of? green pcolor) or
          bush-density > 5)
      [die]
    ]
  ]
end

;; *********************************** Patch Setup ***********************************

to setup-patches
  ask patches [set pcolor 56]
  setup-rocks
  setup-boulders
  setup-puddles
  if (river-toggle = 1) [setup-rivers]
  setup-grass-color
  setup-stockpiles
end

to setup-rocks
  while [count patches with [pcolor < 10] < (max-pycor * max-pxcor * (0.05 + 0.005 * rock-adjustment))] [
    let randxc random-xcor
    let randyc random-ycor
   while [(randyc < max-pycor * stone-centering or randyc > max-pycor * (1 - stone-centering)) or
          (randxc < max-pxcor * stone-centering or randxc > max-pxcor * (1 - stone-centering))]
         [set randxc random-xcor set randyc random-ycor]
    ask patch randxc randyc [
      set pcolor random-rock-color
      ask neighbors [recurse-rocks 100]
    ]
  ]
  ask patches with [pcolor = 10] [
    ifelse (count neighbors with [pcolor < 9] > 2) [set pcolor random-rock-color]
    [set pcolor 56]
  ]
  ask patches with [count neighbors with [pcolor < 10] > count neighbors - 3] [set pcolor random-rock-color]
end

to recurse-rocks [r]
  ifelse (random 120 < r) [
    set pcolor random-rock-color
    ask neighbors with [pcolor > 9] [recurse-rocks r - (random 70)]
  ]
  [set pcolor 10]
end

to setup-boulders
  (foreach range round ( 0.005 * max-pxcor * max-pycor) [
    [index] ->
    let randxc random-xcor
    let randyc random-ycor
    while [([pcolor] of patch randxc randyc < 10)]
          [set randxc random-xcor set randyc random-ycor]
    ask patch randxc randyc [set pcolor random-rock-color boulder-neighbors]
  ])
  ask patches with [count neighbors with [pcolor < 10] > 4] [set pcolor random-rock-color]
end

to boulder-neighbors
  let rand 0
  ask neighbors [
    set rand random 10
    if (rand = 9) [set pcolor random-rock-color ask neighbors4 [set pcolor random-rock-color]]
    if (rand < 2) [set pcolor random-rock-color]
  ]
  ask neighbors
  [
    if ((count neighbors4 with [pcolor < 10] = count neighbors with [pcolor < 10] and
        count neighbors with [pcolor < 10] < 4) or count neighbors with [pcolor < 10] = 1)
       [set pcolor random-rock-color]
  ]
  ask neighbors4 [if (count neighbors4 with [pcolor < 10] >= 2) [set pcolor random-rock-color]]
end

to setup-puddles
  while [count patches with [pcolor = 97] < (max-pycor * max-pxcor * (0.01 + 0.005 * puddle-adjustment))] [
    let randxc random-xcor
    let randyc random-ycor
    while [(randxc > max-pxcor * (5 / 20) and randxc < max-pxcor * (15 / 20)) or
           ([pcolor] of patch randxc randyc < 10)]
          [set randxc random-xcor set randyc random-ycor]
    ask patch randxc randyc [
      set pcolor 97
      ask neighbors [recurse-puddles 60]
    ]
  ]
  ask patches with [pcolor = 10] [
    ifelse (count neighbors with [pcolor = 97] > 2) [set pcolor 97]
    [set pcolor 56]
  ]
end

to recurse-puddles [r]
  ifelse (random 60 < r) [
    set pcolor 97
    ask neighbors with [pcolor = 56] [recurse-puddles r - (20 + random 60)]
  ]
  [set pcolor 10]
end

to setup-rivers
  let startx random-xcor
  while [(startx < max-pxcor / 3 or startx > max-pxcor * (2 / 3))] [set startx random-xcor]
  ask patch startx 0 [
    ask neighbors [set pcolor 97]
    river_recurse 2 5
  ]
  ask patches [
    if (count neighbors with [pcolor = 97] > 2) [set pcolor 97]
  ]
  ask patches [
    if (count neighbors with [pcolor = 97] > 3) [set pcolor 97]
  ]
  ask patches with [pcolor < 10] [if (count neighbors with [pcolor = 97] > 0) [set pcolor 56]]
end

to river_recurse [last-turn momentum]
  set pcolor 97
  ifelse (pxcor != max-pxcor and pycor != max-pycor and pxcor != 0)
  [
    let choice momentum + random 5
    ifelse (choice < 9)
    [
      ifelse (last-turn = 0)
      [ ask patch-at -1 0 [ river_recurse 0 momentum + 1] ]
      [ ask patch-at  1 0 [ river_recurse 1 momentum + 1] ]
    ]
    [ ask patch-at 0 1 [ river_recurse curve-pref-center momentum - 2] ]
  ]
  [ ask neighbors [set pcolor 97] ] ; makes sure end of river is fully connected to edge
end

to setup-stockpiles
  ifelse (team-split-style = 1) [setup-style-1-stockpiles]
  [setup-style-0-and-2-stockpiles]
end

to setup-style-0-and-2-stockpiles
  foreach range stockpile-num [
    let max-low-x max list 0.1 (1 - team-seperation)
    let randxc round ((random (max-pxcor * max-low-x)) + max-pxcor * 0.02)
    let randyc random-ycor
    while [(randyc < max-pycor / 20 or randyc > max-pycor * (19 / 20)) or
           (not shade-of? green [pcolor] of patch randxc randyc) or
           ([count neighbors with [pcolor = red]] of patch randxc randyc > 0)
          ]
          [set randxc round ((random (max-pxcor * max-low-x)) + max-pxcor * 0.02) set randyc random-ycor]
    ask patch randxc randyc [
      set pcolor red
    ]
    let min-high-x max list 0.1 (1 - max-low-x - 0.02)
    set randxc round ((random (max-pxcor * max-low-x)) + max-pxcor * min-high-x)
    set randyc random-ycor
    while [(randyc < max-pycor / 20 or randyc > max-pycor * (19 / 20)) or
           (not shade-of? green [pcolor] of patch randxc randyc) or
           ([count neighbors with [pcolor = violet]] of patch randxc randyc > 0)
          ]
          [set randxc round ((random (max-pxcor * max-low-x)) + max-pxcor * min-high-x) set randyc random-ycor]
    ask patch randxc randyc [
      set pcolor violet
    ]
  ]
end

to setup-style-1-stockpiles
  foreach range stockpile-num [
    let max-low-y max list 0.1 (1 - 0.1 - team-seperation)
    let randyc round ((random (max-pycor * max-low-y)) + max-pycor * 0.02)
    let randxc random-xcor
    while [(randxc < max-pxcor / 20 or randxc > max-pxcor * (19 / 20)) or
           (not shade-of? green [pcolor] of patch randxc randyc) or
           ([count neighbors with [pcolor = red]] of patch randxc randyc > 0)
          ]
          [set randyc round ((random (max-pycor * max-low-y)) + max-pycor * 0.02) set randxc random-xcor]
    ask patch randxc randyc [
      set pcolor red
    ]
    let min-high-y max list 0.1 (1 - max-low-y - 0.02)
    set randyc round ((random (max-pycor * max-low-y)) + max-pycor * min-high-y)
    set randxc random-ycor
    while [(randxc < max-pxcor / 20 or randxc > max-pxcor * (19 / 20)) or
           (not shade-of? green [pcolor] of patch randxc randyc) or
           ([count neighbors with [pcolor = violet]] of patch randxc randyc > 0)
          ]
          [set randyc round ((random (max-pycor * max-low-y)) + max-pycor * min-high-y) set randxc random-xcor]
    ask patch randxc randyc [
      set pcolor violet
    ]
  ]
end

to setup-grass-color
  ask patches with [pcolor = 56]
  [
    if (random 120 = 0)
    [
      let c 56.2
      if (random 2 = 0) [set c 56.4]
      recurse-color c 150
    ]
  ]
end

to recurse-color [c r]
  if (random 120 < r) [
    set pcolor c
    ask neighbors with [pcolor = 56] [recurse-color c r - (30 + random 20)]
  ]
end

;; *********************************** Environment Setup Helpers ***********************************

to-report tree-density
  let ntrees count trees-here
  ask neighbors [set ntrees ntrees + count trees-here]
  report ntrees
end

to-report bush-density
  let nbushes count bushes-here
  ask neighbors [set nbushes nbushes + count bushes-here]
  report nbushes
end

to-report curve-pref-center
  let l-chance 50 + (pxcor - max-pxcor / 2)
  if (random 100 < l-chance) [report 0]
  report 1
end

to-report random-rock-color
  let crand random 10
  if (crand < 1) [report 4.7]
  if (crand < 8) [report 5]
  report 5.3
end

to-report spreadcor [basepoint spread isx?]
  if (isx?) [
    report max (list min list (basepoint + spread / 2 - random-float spread) (max-pxcor - 1) (min-pxcor + 1))
  ]
  report max (list min list (basepoint + spread / 2 - random-float spread) (max-pycor - 1) (min-pycor + 1))
end

to make-halo  [r];; actor procedure
  ;; when you use HATCH, the new turtle inherits the
  ;; characteristics of the parent.  so the halo will
  ;; be the same color as the turtle it encircles (unless
  ;; you add code to change it
  hatch-halos 1
  [ set size r * 3
    ;; Use an RGB color to make halo three fourths transparent
    set color lput 164 extract-rgb [team-color] of myself
    facexy xcor max-pycor
    ;; set thickness of halo to half a patch
    __set-line-thickness 0.1
    ;; We create an invisible directed link from the runner
    ;; to the halo.  Using tie means that whenever the
    ;; runner moves, the halo moves with it.
    create-link-from myself
    [ tie
      hide-link ] ]
end

to change-halos
  ask halos [set size sight-radius * 3]
end

to display-halos
  ask actors with [member? who haloed-actors] [make-halo sight-radius]
end

to hide-halos
  ask halos [die]
end

to highlight-best-actors
  ask max-n-of 5 actors [cumulative_reward] [make-halo 6]
end
@#$#@#$#@
GRAPHICS-WINDOW
12
12
1100
741
-1
-1
12.0
1
10
1
1
1
0
0
0
1
0
89
0
59
0
0
1
ticks
10.0

MONITOR
1150
365
1275
410
red actors
count actors with [team-color = red]
17
1
11

MONITOR
1275
365
1400
410
purple actors
count actors with [team-color = violet]
17
1
11

MONITOR
1150
455
1275
500
red structs
count patches with [shade-of? red pcolor]
17
1
11

MONITOR
1275
455
1400
500
purple structs
count patches with [shade-of? violet pcolor]
17
1
11

MONITOR
1400
365
1620
410
total berries in bushes
sum [berries] of bushes
17
1
11

MONITOR
1400
410
1620
455
total wood in trees
sum [wood] of trees
17
1
11

MONITOR
1150
500
1275
545
NIL
red-bonus
4
1
11

MONITOR
1275
500
1400
545
NIL
purple-bonus
4
1
11

MONITOR
1150
410
1275
455
red-stockpiles
count patches with [pcolor = red]
17
1
11

MONITOR
1275
410
1400
455
purple-stockpiles
count patches with [pcolor = violet]
17
1
11

PLOT
1150
120
1620
360
Structures
time
structs
0.0
500.0
0.0
300.0
false
false
"set-plot-x-range 0 max-ticks" "if (count patches with [shade-of? red pcolor] > plot-y-max - 10 or count patches with [shade-of? violet pcolor] > plot-y-max - 10) [set-plot-y-range 0 (plot-y-max + 50)]"
PENS
"red" 1.0 0 -2674135 true "" "plot count patches with [shade-of? red pcolor]"
"purple" 1.0 0 -8630108 true "" "plot count patches with [shade-of? violet pcolor]"

PLOT
1150
550
1620
770
Living Actors
time
actors
0.0
500.0
0.0
200.0
false
false
"set-plot-x-range 0 max-ticks\nset-plot-y-range 0 (max list sum (sentence team1-pop-dist) sum (sentence team2-pop-dist)) + 5" ""
PENS
"red" 1.0 0 -2674135 true "" "plot count actors with [team-color = red]"
"purple" 1.0 0 -8630108 true "" "plot count actors with [team-color = violet]"

MONITOR
1400
500
1620
545
carried-wood
sum [carried-wood] of actors
17
1
11

MONITOR
1400
455
1620
500
carried-stone
sum [carried-stone] of actors
17
1
11

SLIDER
1265
15
1415
48
sight-radius
sight-radius
1
25
15.0
1
1
NIL
HORIZONTAL

BUTTON
1150
85
1255
118
NIL
change-halos
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
0

INPUTBOX
1265
55
1417
115
haloed-actors
(list 0 30)
1
0
String (reporter)

BUTTON
1150
15
1255
48
NIL
display-halos
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
0

BUTTON
1150
50
1255
83
NIL
hide-halos
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
0

MONITOR
1420
70
1620
115
highest cumulative rewards
sort [precision cumulative_reward 3] of max-n-of (min list (count actors) 5) actors [cumulative_reward]
3
1
11

BUTTON
1420
15
1620
65
NIL
highlight-best-actors
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
0

@#$#@#$#@
## WHAT IS IT?

(a general understanding of what the model is trying to show or explain)

## HOW IT WORKS

(what rules the agents use to create the overall behavior of the model)

## HOW TO USE IT

(how to use the model, including a description of each of the items in the Interface tab)

## THINGS TO NOTICE

(suggested things for the user to notice while running the model)

## THINGS TO TRY

(suggested things for the user to try to do (move sliders, switches, etc.) with the model)

## EXTENDING THE MODEL

(suggested things to add or change in the Code tab to make the model more complicated, detailed, accurate, etc.)

## NETLOGO FEATURES

(interesting or unusual features of NetLogo that the model uses, particularly in the Code tab; or where workarounds were needed for missing features)

## RELATED MODELS

(models in the NetLogo Models Library and elsewhere which are of related interest)

## CREDITS AND REFERENCES

(a reference to the model's URL on the web if it has one, as well as any other necessary credits, citations, and links)
@#$#@#$#@
default
true
0
Polygon -7500403 true true 150 5 40 250 150 205 260 250

airplane
true
0
Polygon -7500403 true true 150 0 135 15 120 60 120 105 15 165 15 195 120 180 135 240 105 270 120 285 150 270 180 285 210 270 165 240 180 180 285 195 285 165 180 105 180 60 165 15

arrow
true
0
Polygon -7500403 true true 150 0 0 150 105 150 105 293 195 293 195 150 300 150

box
false
0
Polygon -7500403 true true 150 285 285 225 285 75 150 135
Polygon -7500403 true true 150 135 15 75 150 15 285 75
Polygon -7500403 true true 15 75 15 225 150 285 150 135
Line -16777216 false 150 285 150 135
Line -16777216 false 150 135 15 75
Line -16777216 false 150 135 285 75

bug
true
0
Circle -7500403 true true 96 182 108
Circle -7500403 true true 110 127 80
Circle -7500403 true true 110 75 80
Line -7500403 true 150 100 80 30
Line -7500403 true 150 100 220 30

bush
true
0
Rectangle -7500403 true true 60 90 240 210
Circle -7500403 true true 45 75 60
Circle -7500403 true true 69 189 42
Circle -7500403 true true 39 174 42
Circle -7500403 true true 210 105 60
Circle -7500403 true true 195 75 60
Circle -7500403 true true 240 150 30
Circle -7500403 true true 180 75 30
Circle -7500403 true true 195 150 60
Circle -7500403 true true 129 189 42
Circle -7500403 true true 204 189 42
Circle -7500403 true true 174 204 42
Circle -7500403 true true 159 189 42
Circle -7500403 true true 99 204 42
Circle -7500403 true true 54 144 42
Circle -7500403 true true 39 114 42
Circle -7500403 true true 99 69 42
Circle -7500403 true true 129 84 42
Circle -7500403 true true 159 84 42
Circle -13791810 true false 90 105 30
Circle -13791810 true false 135 165 30
Circle -13791810 true false 180 120 30
Circle -13791810 true false 210 180 30
Circle -13791810 true false 60 180 30

butterfly
true
0
Polygon -7500403 true true 150 165 209 199 225 225 225 255 195 270 165 255 150 240
Polygon -7500403 true true 150 165 89 198 75 225 75 255 105 270 135 255 150 240
Polygon -7500403 true true 139 148 100 105 55 90 25 90 10 105 10 135 25 180 40 195 85 194 139 163
Polygon -7500403 true true 162 150 200 105 245 90 275 90 290 105 290 135 275 180 260 195 215 195 162 165
Polygon -16777216 true false 150 255 135 225 120 150 135 120 150 105 165 120 180 150 165 225
Circle -16777216 true false 135 90 30
Line -16777216 false 150 105 195 60
Line -16777216 false 150 105 105 60

car
false
0
Polygon -7500403 true true 300 180 279 164 261 144 240 135 226 132 213 106 203 84 185 63 159 50 135 50 75 60 0 150 0 165 0 225 300 225 300 180
Circle -16777216 true false 180 180 90
Circle -16777216 true false 30 180 90
Polygon -16777216 true false 162 80 132 78 134 135 209 135 194 105 189 96 180 89
Circle -7500403 true true 47 195 58
Circle -7500403 true true 195 195 58

circle
false
0
Circle -7500403 true true 0 0 300

circle 2
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240

cow
false
0
Polygon -7500403 true true 200 193 197 249 179 249 177 196 166 187 140 189 93 191 78 179 72 211 49 209 48 181 37 149 25 120 25 89 45 72 103 84 179 75 198 76 252 64 272 81 293 103 285 121 255 121 242 118 224 167
Polygon -7500403 true true 73 210 86 251 62 249 48 208
Polygon -7500403 true true 25 114 16 195 9 204 23 213 25 200 39 123

cylinder
false
0
Circle -7500403 true true 0 0 300

default-plus
true
0
Polygon -7500403 true true 150 5 40 250 150 205 260 250
Line -16777216 false 150 15 240 225
Line -16777216 false 150 15 60 225

dot
false
0
Circle -7500403 true true 90 90 120

face happy
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 255 90 239 62 213 47 191 67 179 90 203 109 218 150 225 192 218 210 203 227 181 251 194 236 217 212 240

face neutral
false
0
Circle -7500403 true true 8 7 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Rectangle -16777216 true false 60 195 240 225

face sad
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 168 90 184 62 210 47 232 67 244 90 220 109 205 150 198 192 205 210 220 227 242 251 229 236 206 212 183

fish
false
0
Polygon -1 true false 44 131 21 87 15 86 0 120 15 150 0 180 13 214 20 212 45 166
Polygon -1 true false 135 195 119 235 95 218 76 210 46 204 60 165
Polygon -1 true false 75 45 83 77 71 103 86 114 166 78 135 60
Polygon -7500403 true true 30 136 151 77 226 81 280 119 292 146 292 160 287 170 270 195 195 210 151 212 30 166
Circle -16777216 true false 215 106 30

flag
false
0
Rectangle -7500403 true true 60 15 75 300
Polygon -7500403 true true 90 150 270 90 90 30
Line -7500403 true 75 135 90 135
Line -7500403 true 75 45 90 45

flower
false
0
Polygon -10899396 true false 135 120 165 165 180 210 180 240 150 300 165 300 195 240 195 195 165 135
Circle -7500403 true true 85 132 38
Circle -7500403 true true 130 147 38
Circle -7500403 true true 192 85 38
Circle -7500403 true true 85 40 38
Circle -7500403 true true 177 40 38
Circle -7500403 true true 177 132 38
Circle -7500403 true true 70 85 38
Circle -7500403 true true 130 25 38
Circle -7500403 true true 96 51 108
Circle -16777216 true false 113 68 74
Polygon -10899396 true false 189 233 219 188 249 173 279 188 234 218
Polygon -10899396 true false 180 255 150 210 105 210 75 240 135 240

house
false
0
Rectangle -7500403 true true 45 120 255 285
Rectangle -16777216 true false 120 210 180 285
Polygon -7500403 true true 15 120 150 15 285 120
Line -16777216 false 30 120 270 120

leaf
false
0
Polygon -7500403 true true 150 210 135 195 120 210 60 210 30 195 60 180 60 165 15 135 30 120 15 105 40 104 45 90 60 90 90 105 105 120 120 120 105 60 120 60 135 30 150 15 165 30 180 60 195 60 180 120 195 120 210 105 240 90 255 90 263 104 285 105 270 120 285 135 240 165 240 180 270 195 240 210 180 210 165 195
Polygon -7500403 true true 135 195 135 240 120 255 105 255 105 285 135 285 165 240 165 195

line
true
0
Line -7500403 true 150 0 150 300

line half
true
0
Line -7500403 true 150 0 150 150

pentagon
false
0
Polygon -7500403 true true 150 15 15 120 60 285 240 285 285 120

person
false
0
Circle -7500403 true true 110 5 80
Polygon -7500403 true true 105 90 120 195 90 285 105 300 135 300 150 225 165 300 195 300 210 285 180 195 195 90
Rectangle -7500403 true true 127 79 172 94
Polygon -7500403 true true 195 90 240 150 225 180 165 105
Polygon -7500403 true true 105 90 60 150 75 180 135 105

person42
false
0
Circle -13791810 true false 110 5 80
Polygon -13791810 true false 105 90 120 195 90 285 105 300 135 300 150 225 165 300 195 300 210 285 180 195 195 90
Rectangle -13791810 true false 127 79 172 94
Polygon -13791810 true false 195 90 240 150 225 180 165 105
Polygon -13791810 true false 105 90 60 150 75 180 135 105

plant
false
0
Rectangle -7500403 true true 135 90 165 300
Polygon -7500403 true true 135 255 90 210 45 195 75 255 135 285
Polygon -7500403 true true 165 255 210 210 255 195 225 255 165 285
Polygon -7500403 true true 135 180 90 135 45 120 75 180 135 210
Polygon -7500403 true true 165 180 165 210 225 180 255 120 210 135
Polygon -7500403 true true 135 105 90 60 45 45 75 105 135 135
Polygon -7500403 true true 165 105 165 135 225 105 255 45 210 60
Polygon -7500403 true true 135 90 120 45 150 15 180 45 165 90

sheep
false
15
Circle -1 true true 203 65 88
Circle -1 true true 70 65 162
Circle -1 true true 150 105 120
Polygon -7500403 true false 218 120 240 165 255 165 278 120
Circle -7500403 true false 214 72 67
Rectangle -1 true true 164 223 179 298
Polygon -1 true true 45 285 30 285 30 240 15 195 45 210
Circle -1 true true 3 83 150
Rectangle -1 true true 65 221 80 296
Polygon -1 true true 195 285 210 285 210 240 240 210 195 210
Polygon -7500403 true false 276 85 285 105 302 99 294 83
Polygon -7500403 true false 219 85 210 105 193 99 201 83

square
false
0
Rectangle -7500403 true true 30 30 270 270

square 2
false
0
Rectangle -7500403 true true 30 30 270 270
Rectangle -16777216 true false 60 60 240 240

star
false
0
Polygon -7500403 true true 151 1 185 108 298 108 207 175 242 282 151 216 59 282 94 175 3 108 116 108

target
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240
Circle -7500403 true true 60 60 180
Circle -16777216 true false 90 90 120
Circle -7500403 true true 120 120 60

thin square
true
0
Rectangle -7500403 false true 45 45 255 255

tree
false
0
Circle -7500403 true true 118 3 94
Rectangle -6459832 true false 120 195 180 300
Circle -7500403 true true 65 21 108
Circle -7500403 true true 116 41 127
Circle -7500403 true true 45 90 120
Circle -7500403 true true 104 74 152

triangle
false
0
Polygon -7500403 true true 150 30 15 255 285 255

triangle 2
false
0
Polygon -7500403 true true 150 30 15 255 285 255
Polygon -16777216 true false 151 99 225 223 75 224

truck
false
0
Rectangle -7500403 true true 4 45 195 187
Polygon -7500403 true true 296 193 296 150 259 134 244 104 208 104 207 194
Rectangle -1 true false 195 60 195 105
Polygon -16777216 true false 238 112 252 141 219 141 218 112
Circle -16777216 true false 234 174 42
Rectangle -7500403 true true 181 185 214 194
Circle -16777216 true false 144 174 42
Circle -16777216 true false 24 174 42
Circle -7500403 false true 24 174 42
Circle -7500403 false true 144 174 42
Circle -7500403 false true 234 174 42

turtle
true
0
Polygon -10899396 true false 215 204 240 233 246 254 228 266 215 252 193 210
Polygon -10899396 true false 195 90 225 75 245 75 260 89 269 108 261 124 240 105 225 105 210 105
Polygon -10899396 true false 105 90 75 75 55 75 40 89 31 108 39 124 60 105 75 105 90 105
Polygon -10899396 true false 132 85 134 64 107 51 108 17 150 2 192 18 192 52 169 65 172 87
Polygon -10899396 true false 85 204 60 233 54 254 72 266 85 252 107 210
Polygon -7500403 true true 119 75 179 75 209 101 224 135 220 225 175 261 128 261 81 224 74 135 88 99

wheel
false
0
Circle -7500403 true true 3 3 294
Circle -16777216 true false 30 30 240
Line -7500403 true 150 285 150 15
Line -7500403 true 15 150 285 150
Circle -7500403 true true 120 120 60
Line -7500403 true 216 40 79 269
Line -7500403 true 40 84 269 221
Line -7500403 true 40 216 269 79
Line -7500403 true 84 40 221 269

wolf
false
0
Polygon -16777216 true false 253 133 245 131 245 133
Polygon -7500403 true true 2 194 13 197 30 191 38 193 38 205 20 226 20 257 27 265 38 266 40 260 31 253 31 230 60 206 68 198 75 209 66 228 65 243 82 261 84 268 100 267 103 261 77 239 79 231 100 207 98 196 119 201 143 202 160 195 166 210 172 213 173 238 167 251 160 248 154 265 169 264 178 247 186 240 198 260 200 271 217 271 219 262 207 258 195 230 192 198 210 184 227 164 242 144 259 145 284 151 277 141 293 140 299 134 297 127 273 119 270 105
Polygon -7500403 true true -1 195 14 180 36 166 40 153 53 140 82 131 134 133 159 126 188 115 227 108 236 102 238 98 268 86 269 92 281 87 269 103 269 113

x
false
0
Polygon -7500403 true true 270 75 225 30 30 225 75 270
Polygon -7500403 true true 30 75 75 30 270 225 225 270
@#$#@#$#@
NetLogo 6.1.0
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
default
0.0
-0.2 0 0.0 1.0
0.0 1 1.0 0.0
0.2 0 0.0 1.0
link direction
true
0
Line -7500403 true 150 150 90 180
Line -7500403 true 150 150 210 180
@#$#@#$#@
1
@#$#@#$#@
