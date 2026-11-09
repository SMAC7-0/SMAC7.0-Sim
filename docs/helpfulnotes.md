# Make a visible point marker
<site
    name="top_joint_marker"
    pos="0 0 0"
    size="0.01"
    rgba="1 0 0 1"
/>





# Initial joints set
<mujoco>

  <asset>
    <mesh name="ee1_mesh"   file="ee1-inchworm.stl"   scale="0.001 0.001 0.001"/>
    <mesh name="ee2_mesh"   file="ee2-inchworm.stl"   scale="0.001 0.001 0.001"/>
    <mesh name="link1_mesh" file="link1-inchworm.stl" scale="0.001 0.001 0.001"/>
    <mesh name="link2_mesh" file="link2-inchworm.stl" scale="0.001 0.001 0.001"/>
  </asset>

  <default>
    <joint
      damping="0.05"
      armature="0.001"
      range="-1.6 1.6"
    />
    <geom
      contype="0"
      conaffinity="0"
      rgba="0.7 0.7 0.75 1"
    />
  </default>

  <worldbody>
    <body name="ee1" pos="0 0 -0.05">
      <freejoint name="root"/>
      <geom
        type="mesh"
        mesh="ee1_mesh"
        pos="0.0644 0 0.0712"
        euler="0 0 0"
        mass="0.05"
      />
      <body name="link1" pos="0 0 0">
        <joint
          name="left_wrist"
          type="hinge"
          axis="0 0 1"
          pos="0.11 0.011 0"
        />
        <joint
          name="left_hand"
          type="hinge"
          axis="0 1 0"
          pos="0.105 0 0.1855"
        />
        <geom
          type="mesh"
          mesh="link1_mesh"
          pos="0.0269 0 0.1675"
          euler="0 0 0"
          mass="0.05"
        />

        <body name="link2" pos="0 0 0">
          <joint
            name="top"
            type="hinge"
            axis="0 1 0"
            pos="0.0401 0 0.339"
          />
          <geom
            type="mesh"
            mesh="link2_mesh"
            pos="-0.0444 0 0.1655"
            euler="0 0 0"
            mass="0.05"
          />

          <body name="ee2" pos="0 0 0">
            <joint
              name="right_hand"
              type="hinge"
              axis="0 1 0"
              pos="0 0 0.18"
            />
            <joint
              name="right_wrist"
              type="hinge"
              axis="0 0 1"
              pos="0 0.011 0"
            />
            <geom
              type="mesh"
              mesh="ee2_mesh"
              pos="-0.0439 0 0.0702"
              euler="0 0 0"
              mass="0.05"
            />

          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <position
      name="left_wrist_act"
      joint="left_wrist"
      kp="2"
    />
    <position
      name="left_hand_act"
      joint="left_hand"
      kp="2"
    />
    <position
      name="top_act"
      joint="top"
      kp="2"
    />
    <position
      name="right_hand_act"
      joint="right_hand"
      kp="2"
    />
    <position
      name="right_wrist_act"
      joint="right_wrist"
      kp="2"
    />
  </actuator>

</mujoco>