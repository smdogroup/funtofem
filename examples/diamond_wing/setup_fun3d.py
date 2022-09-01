import caps2fun

caps_fluid = caps2fun.CapsFluid.default(csmFile="diamondWing.csm")
pointwise_aim = caps_fluid.pointwiseAim
fun3d_aim = caps_fluid.fun3dAim

pointwise_aim.set_mesh(
        inviscid=True
)
fun3d_aim.flow_settings = caps2fun.FlowSettings(
        flow_type="inviscid",
        mach_number=0.3,
        angle_of_attack=0.0,
        reynolds_number=1e6,
        temperature_ref=300.0,
        ref_area=1.0,
        num_steps=10,
        freeze_limiter_iteration=None
        )
fun3d_aim.motion_settings = caps2fun.MotionSettings(body_name="diamondWing")
fun3d_aim.build_complex = True

caps_fun3d = caps2fun.CapsFun3d(pointwise_aim=pointwise_aim, fun3d_aim=fun3d_aim)
caps_fun3d.build_mesh()
caps_fun3d.prepare_fun3d()